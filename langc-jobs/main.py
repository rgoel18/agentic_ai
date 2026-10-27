from dotenv import load_dotenv
load_dotenv()

import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
#from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from huggingface_hub import InferenceClient

from openai import OpenAI
import os


# Fetch LangGraph version safely from environment metadata
try:
    lg_version = metadata.version("langgraph")
except metadata.PackageNotFoundError:
    lg_version = "Not Installed"

print(f"langchain version: {core_version}")

print(f"langgraph version: {lg_version}")


# Initialize the client with your free API token
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    print("Hello from langc-jobs!")
    #llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    #client = InferenceClient(token=HF_TOKEN)
    #response = llm.invoke("Say0 setup complete")
    # Call a text generation model
    #response = client.chat_completion(
    #model="meta-llama/Llama-3.2-3B-Instruct",
    #messages=[
    #    {
    #        "role": "user",
    #        "content": "Explain the difference between a list and a tuple in Python."
    #    }
    #],
    #max_tokens=150,

 
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,   # the token you just created
    )
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",   # or any open model on the router
        #openai/gpt-oss-20b
        #qwen/qwen3.6-27b
        messages=[{"role": "user", "content": "Explain Airflow DAGs in one line"}],
    )
    print(resp.choices[0].message.content)
    #print(f"response:{resp}")


    #print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
