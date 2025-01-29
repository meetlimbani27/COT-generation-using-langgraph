from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END

class MedicalState(TypedDict):
    patient_detail: str
    doctor_reply: str
    messages: List[Dict[str, str]]
    current_speaker: Literal["patient", "doctor", "resolution_check"]
    resolved: bool

class MedicalAgents:
    def __init__(self):
        self.patient_model = ChatOpenAI(model="gpt-4o")
        self.doctor_model = ChatOpenAI(model="gpt-4o")
        self.resolver_model = ChatOpenAI(model="gpt-4o")

    def patient_node(self, state: MedicalState):
        if len(state["messages"]) == 0:
            return {
                "messages": [{"role": "patient", "content": state["patient_detail"]}],
                "current_speaker": "patient"
            }
        
        prompt = f"""As a patient with this initial concern: {state['patient_detail']}
        Continue the conversation based on the doctor's last response: 
        {state['messages'][-1]['content']}"""
        response = self.patient_model.invoke(prompt)
        return {
            "messages": state["messages"] + [{"role": "patient", "content": response.content}],
            "current_speaker": "patient"
        }

    def doctor_node(self, state: MedicalState):
        prompt = f"""You are a homeopathic doctor. Provide specific remedies and practical advice. 
        Only suggest professional consultation if absolutely necessary. 
        Example response style: {state['doctor_reply']}
        
        Patient's history: {state['patient_detail']}
        Conversation history: {[m['content'] for m in state['messages']]}"""
        
        response = self.doctor_model.invoke(prompt)
        return {
            "messages": state["messages"] + [{"role": "doctor", "content": response.content}],
            "current_speaker": "doctor"
        }

    def resolution_node(self, state: MedicalState):
        if len(state["messages"]) >= 8:  # Fallback to prevent infinite loops
            return {"resolved": True}
        
        prompt = f"""Determine if the patient's query is fully resolved:
        
        Initial query: {state['patient_detail']}
        Conversation: {" ".join([m['content'] for m in state['messages']])}
        
        Has the doctor provided:
        1. Specific remedy/recommendation?
        2. Clear application instructions?
        3. Avoided unnecessary referrals?
        
        Answer only 'yes' or 'no'."""
        
        response = self.resolver_model.invoke(prompt)
        resolved = "yes" in response.content.lower()
        return {"resolved": resolved}

def should_continue(state: MedicalState) -> str:
    if state["resolved"]:
        return END
    return "patient" if state["current_speaker"] == "doctor" else "doctor"

def build_workflow():
    workflow = StateGraph(MedicalState)
    agents = MedicalAgents()
    
    workflow.add_node("patient", agents.patient_node)
    workflow.add_node("doctor", agents.doctor_node)
    workflow.add_node("resolution_check", agents.resolution_node)
    
    workflow.set_entry_point("patient")
    
    workflow.add_edge("patient", "doctor")
    workflow.add_edge("doctor", "resolution_check")
    
    workflow.add_conditional_edges(
        "resolution_check",
        lambda s: END if s["resolved"] else "patient",
    )
    
    return workflow

def generate_cot(row):
    workflow = build_workflow()
    app = workflow.compile()
    
    initial_state = {
        "patient_detail": row["patient_detail"],
        "doctor_reply": row["doctor_reply"],
        "messages": [],
        "current_speaker": "patient",
        "resolved": False
    }
    
    messages = []
    for event in app.stream(initial_state):
        for key, value in event.items():
            if "messages" in value:
                messages = value["messages"]
            if value.get("resolved"):
                break
    
    # Save conversation and return results
    with open("conversation_log.txt", "a") as f:
        f.write("\n----- New Conversation -----\n")
        for msg in messages:
            f.write(f"{msg['role'].title()}: {msg['content']}\n\n")
    
    return {
        "question": row["question"],
        "cot": [msg["content"] for msg in messages]
    }

# Process single row
sample_row = {
    "question": """Q. Kindly suggest a homeopathic medicine to stop hairfall and promote hair growth.""",
    "patient_detail": """Hello doctor,
I am 24 years old, and for the past nine years, I am facing hair fall problem. Nowadays, my 60 % of hair is falling on my top and front of my head. I checked my thyroid and hemoglobin many times, but their reports are good. I use many home remedies and hair oils, but when I stop using it, again, it starts falling. The allopathic doctor says use Minoxidil 5 %, Finasteride, and hair serum. But Finasteride has many side effects. Can you please tell me is any medicine available in homeopathy which stops hair fall and promotes hair growth? If yes, can you please tell me the name?""",
    "doctor_reply": """Hello. I checked the attached photo (attachment removed to protect patient identity) and read your description. It seems you have been suffering from hair loss problem for a longtime. Do you eat healthy food like vegetables and fruits every day? Sometimes lack of nutrition is also the reason for hair loss. Do you know if your father also differed from hair loss issue at this young age? Did you suffer from severe health issues or chronic illnesses? If you do not think above mentioned is the cause for your hair fall, then the only reason I can think is that you have been going through severe stress which has taken a toll on your health. Homeopathy would be a good option in this case as it will cure our problem from the root cause, and there will be no side effects. So to prescribe you a correct homeopathic remedy I need to understand your mental, emotional, and physical state. He ce I need detail case history which can be done either through face to face consultation or online consultation. I would advise you to visit a good homeopath for consultation. For the time being, you can apply arnica hair oil twice a week on your hair. Just mix one spoon of Arnica oil with five spoons of coconut oil and apply all over your scalp. You can gently massage your ear with this mixture. Try this for 15 days and then let me know how you feel. I hope you start feeling better. Let me know if you have any questions.
"""
}

result = generate_cot(sample_row)
print("Generated COT:")
for i, message in enumerate(result["cot"]):
    speaker = "Patient" if i % 2 == 0 else "Doctor"
    print(f"{speaker}: {message}")

print(f"\nConversation stored in 'conversation_log.txt'")