import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def test_single_chunk(chunk_text: str):
    """
    Generate 10 advanced, thought-provoking questions from a single chunk of text
    to test the prompt and question-generation flow.
    """
    # 1. Initialize LLM
    openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=openai_api_key,
        temperature=0.7
    )

    # 2. Prepare the prompt
    prompt_text = f"""
You are given the following medical text. Please produce 10 simple, multi-step
questions that require deeper reasoning or interpretation to understand the concept, encouraging a chain
of thought from a specialized reasoning model. Ensure each question is relevant
to the text and has practical value for someone from non-medical background seeking to understand it at a
high level. Return exactly 10 questions, each on a new line, with no additional
explanation or numbering.

Medical Text:
\"\"\"{chunk_text}\"\"\"

Questions:
"""

    # 3. Generate the questions (async)
    messages = [HumanMessage(content=prompt_text)]
    response = await llm.ainvoke(messages)
    text_output = response.content
    
    print(text_output)

async def main():
    """
    Main entry point for testing question generation with a single chunk.
    Replace 'example_chunk_text' with your actual medical text.
    """
    example_chunk_text = """
acids, while nucleases catalyze the hydrolysis of the phosphoester bonds in DNA and RNA. 
Careful control of the activities of these enzymes is required to ensure that they act only 
on appropriate target molecules at appropriate times. 

Many Metabolic Reactions Involve Group Transfer
Many of the enzymic reactions responsible for synthesis and breakdown of biomolecules 
involve the transfer of a chemical group G from a donor D to an acceptor A to form an 
acceptor group complex, A—G: The hydrolysis and phosphorolysis of glycogen, for example, 
involve the transfer of glucosyl groups to water or to orthophosphate. 

The equilibrium constant for the hydrolysis of covalent bonds strongly favors the formation 
of split products. Conversely, in many cases the group transfer reactions responsible for 
the biosynthesis of macromolecules involve the thermodynamically unfavored formation of 
covalent bonds. Enzyme catalysts play a critical role in surmounting these barriers by 
virtue of their capacity to
"""
    await test_single_chunk(example_chunk_text)

if __name__ == "__main__":
    asyncio.run(main())
