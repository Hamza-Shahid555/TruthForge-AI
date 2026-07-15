# verify_env.py — run this from D:\new agent, with your venv activated
import os
from dotenv import load_dotenv

load_dotenv()  # no path argument -> loads ".env" from the current directory by default

key = os.getenv("OPENAI_API_KEY")
project = os.getenv("LANGCHAIN_PROJECT")

if key:
    print(f"OPENAI_API_KEY loaded, starts with: {key[:7]}...")
else:
    print("OPENAI_API_KEY not found — check .env exists in this folder")

print(f"LANGCHAIN_PROJECT = {project}")

import importlib.metadata as importlib_metadata

try:
    langgraph_version = importlib_metadata.version("langgraph")
except importlib_metadata.PackageNotFoundError:
    langgraph_version = "unavailable"

print(f"langgraph version = {langgraph_version}")