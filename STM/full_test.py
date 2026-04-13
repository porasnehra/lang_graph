#!/usr/bin/env python3
"""
LangGraph with Postgres Stateful Memory Test
This script demonstrates LangGraph persistence using PostgreSQL
"""

import sys
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URI = "postgresql://postgres:postgres@localhost:5450/postgres"
MODEL = "gemini-2.5-flash"

def main():
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            api_key=API_KEY,
            streaming=True
        )

    # 3. Define state and model function
    class MessagesState(TypedDict):
        messages: Annotated[list, add_messages]
   
    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    try:
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")

    except Exception as e:
        print(f" Graph build failed: {e}")
        return False
    
    try:
        print(f"\nConnecting to Postgres: {DB_URI.split('@')[1]}")
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            
            checkpointer.setup()
            
            graph = builder.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": "1"}}
            
            result1 = graph.invoke(
                {"messages": [("user", "Hi, I'm Poras")]},
                config
            )
            print(f"Assistant: {result1['messages'][-1].content}")
            
            result2 = graph.invoke(
                {"messages": [("user", "What is my name?")]},
                config
            )
            print(f"Assistant: {result2['messages'][-1].content}")
            
            config2 = {"configurable": {"thread_id": "2"}}
            
            result3 = graph.invoke(
                {"messages": [("user", "Hi, I'm Alice")]},
                config2
            )
            print(f"Assistant: {result3['messages'][-1].content}")
            
    except Exception as e:
        print(f"Postgres connection or execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)