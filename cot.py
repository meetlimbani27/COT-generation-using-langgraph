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
        1. Specific homeopathic remedy/recommendation?
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

# Example usage remains the same