import os
import asyncio
import asyncpg
from langchain.llms import OpenAI

# -----------------------------------------------------------------------------
# 1. Database Connection (async)
# -----------------------------------------------------------------------------
async def get_db_connection():
    """
    Establish an async connection to the PostgreSQL database using asyncpg.
    Credentials should ideally be stored in environment variables for security.
    """
    dbname = os.getenv("DB_NAME", "your_db_name")
    user = os.getenv("DB_USER", "your_username")
    password = os.getenv("DB_PASS", "your_password")
    host = os.getenv("DB_HOST", "your_host")
    port = os.getenv("DB_PORT", "your_port")

    connection = await asyncpg.connect(
        user=user,
        password=password,
        database=dbname,
        host=host,
        port=port
    )
    return connection

# -----------------------------------------------------------------------------
# 2. Fetch Data (async)
# -----------------------------------------------------------------------------
async def fetch_chunks(conn, table_name="your_table_name"):
    """
    Fetch chunk data from the specified table asynchronously.
    Expects two columns: 'id' and 'chunk_text'.
    """
    rows = await conn.fetch(f"SELECT id, chunk_text FROM {table_name}")
    # Convert asyncpg Records into a list of tuples
    chunks = [(row["id"], row["chunk_text"]) for row in rows]
    return chunks

# -----------------------------------------------------------------------------
# 3. Initialize LLM (synchronous instantiation, async usage)
# -----------------------------------------------------------------------------
def initialize_llm():
    """
    Initialize the OpenAI language model from LangChain.
    The API key should be set as an environment variable for security.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    llm = OpenAI(
        model_name="gpt-4",
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    return llm

# -----------------------------------------------------------------------------
# 4. Generate Questions (async)
# -----------------------------------------------------------------------------
async def generate_questions(llm, chunk):
    """
    Generate 5 advanced, thought-provoking questions from the given chunk of text.
    These questions should encourage a specialized reasoning model to produce
    detailed, chain-of-thought answers, but still be understandable for a lay audience.
    """

    prompt = f"""
You are given the following medical text. Please produce 5 advanced individual
questions that require deeper reasoning or interpretation, encouraging a chain
of thought from a specialized reasoning model. Ensure each question is relevant
to the text and has practical value for someone seeking to understand it at a
high level. Return exactly 5 questions, each on a new line, with no additional
explanation or numbering.

Medical Text:
\"\"\"{chunk}\"\"\"

Questions:
"""

    # Use the async generation call from LangChain
    response = await llm.agenerate([prompt])
    # The response is a list of LLMResult, we only sent a single prompt
    text_output = response.generations[0][0].text

    # Split by newlines, remove empty lines, and ensure exactly 5 items
    lines = [line.strip() for line in text_output.split("\n") if line.strip()]
    questions = lines[:5]  # if there are more than 5, truncate

    return questions

# -----------------------------------------------------------------------------
# 5. Insert Generated Questions (async)
# -----------------------------------------------------------------------------
async def insert_questions(conn, chunk_id, questions, table_name="generated_questions"):
    """
    Insert the generated questions into the database asynchronously.
    Assumes a table schema: (id SERIAL PRIMARY KEY, chunk_id INTEGER, question TEXT).
    """
    for question in questions:
        await conn.execute(
            f"INSERT INTO {table_name} (chunk_id, question) VALUES ($1, $2)",
            chunk_id,
            question
        )

# -----------------------------------------------------------------------------
# 6. Main Async Flow
# -----------------------------------------------------------------------------
async def main():
    # 6a. Connect to the database
    conn = await get_db_connection()

    # 6b. Fetch the chunks from the database
    chunks = await fetch_chunks(conn, table_name="your_table_name")

    # 6c. Initialize the LLM
    llm = initialize_llm()

    # 6d. Iterate over chunks, generate, and insert questions
    for chunk_id, chunk_text in chunks:
        questions = await generate_questions(llm, chunk_text)
        await insert_questions(conn, chunk_id, questions, table_name="generated_questions")

    # 6e. Close the database connection
    await conn.close()

# -----------------------------------------------------------------------------
# 7. Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
