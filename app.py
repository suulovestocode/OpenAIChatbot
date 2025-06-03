import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from dateutil.parser import parse as date_parse
from typing import TypedDict, Any, List

# Load environment variables
dotenv_path = os.getenv("DOTENV_PATH", ".env")
load_dotenv(dotenv_path)

# PostgreSQL connection URL
POSTGRES_URL = (
    f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)

# LLM setup
llm = ChatOpenAI(
    model="llama-3.1-70b-fireworks",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# State schema for LangGraph
class GraphState(TypedDict, total=False):
    table_name: str
    columns: List[str]
    latest_result: Any
    user_question: str
    sql_query: str

# Helper: auto-detect and convert date columns
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(5)
        if sum(1 for v in sample if "-" in v or "/" in v) >= 4:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True).dt.date
            except:
                pass
    return df

# Node 1: ingest CSV into PostgreSQL
def ingest_data(state: GraphState) -> GraphState:
    file = st.session_state["uploaded_file"]
    df = pd.read_csv(file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = convert_dates(df)

    table_name = os.path.splitext(file.name)[0].lower()
    engine = create_engine(POSTGRES_URL)
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    # Retrieve column list and types
    schema_rows = engine.connect().execute(
        text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'")
    ).fetchall()
    cols = [row[0] for row in schema_rows]
    types = {row[0]: row[1] for row in schema_rows}

    state["table_name"] = table_name
    state["columns"] = cols
    state["types"] = types
    return state

# Node 2: generate SQL, adjust boolean/int mismatches, run it, and return DataFrame
def llm_execute_query(state: GraphState) -> GraphState:
    cols = ", ".join(state.get("columns", []))
    prompt = f"""
You are an expert in PostgreSQL. The table is `{state['table_name']}` with columns: {cols}.
User question: {state['user_question']}
-Always refer to the database and do not conjure up your own columns
DO NOT SORT IN DESCENDING OR ASCENDING
Provide exactly one valid SQL SELECT query (no explanation, no markdown).
"""
    raw = llm.invoke(prompt).content.strip()
    # Remove code fences if any
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
    # Strip stray labels
    raw = re.sub(r'^(?:sql\s*[:]?|~sql\s*[:]?)+', '', raw, flags=re.IGNORECASE).strip()
    # Find the SQL starting point
    m = re.search(r"\b(SELECT|WITH)\b", raw, flags=re.IGNORECASE)
    sql_query = raw[m.start():] if m else raw

    # Fix boolean comparisons for integer flag columns
    for col, dtype in state.get("types", {}).items():
        if dtype in ("bigint", "integer") and col.endswith("_flag"):
            # replace = TRUE/FALSE with = 1/0
            sql_query = re.sub(rf"\b{col}\s*=\s*TRUE\b", f"{col} = 1", sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(rf"\b{col}\s*=\s*FALSE\b", f"{col} = 0", sql_query, flags=re.IGNORECASE)
    # Also handle generic literals
    sql_query = re.sub(r"=\s*TRUE\b", "= 1", sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r"=\s*FALSE\b", "= 0", sql_query, flags=re.IGNORECASE)

    # Execute SQL into a DataFrame
    engine = create_engine(POSTGRES_URL)
    df_result = pd.read_sql_query(sql_query, con=engine)

    state["latest_result"] = df_result
    state["sql_query"] = sql_query
    return state

# Build LangGraph
graph = StateGraph(GraphState)
graph.add_node("ingest_data", ingest_data)
graph.add_node("llm_execute_query", llm_execute_query)
# Entry and exit
graph.set_entry_point("ingest_data")
graph.add_edge("ingest_data", "llm_execute_query")
graph.set_finish_point("llm_execute_query")
chatbot = graph.compile()

# Streamlit UI
st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("AI-driven SQL Chatbot")

# CSV Upload
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    st.session_state["uploaded_file"] = uploaded
    st.success("CSV uploaded. Ask your query below.")

# Query input and execution
if "uploaded_file" in st.session_state:

    # wrap in a form so Enter == submit
    with st.form(key="query_form"):
        q = st.text_input("Ask your question:", key="query_input")
        submitted = st.form_submit_button("Run Query")

    if submitted and q:
        with st.spinner("Running…"):
            init_state: GraphState = {
                "table_name": "",
                "columns": [],
                "types": {},
                "latest_result": None,
                "user_question": q,
                "sql_query": ""
            }
            out = chatbot.invoke(init_state)

        # Show SQL
        st.subheader("🔍 Generated SQL Query")
        st.code(out["sql_query"], language="sql")
        # Show result table
        st.subheader("📊 Query Result")
        st.dataframe(out["latest_result"])