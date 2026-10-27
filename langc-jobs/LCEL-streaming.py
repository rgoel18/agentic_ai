from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def demo_streaming():
    #Demonstrate streaming for real-time output.
    prompt = ChatPromptTemplate.from_template("Write a 10 haiku about: {topic}")
    model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    streaming=True,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    #reasoning_effort="low",
    max_tokens=8000, 
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    #Streaming: run with streaming enabled
    for chunk in chain.stream({"topic":"nature"}):
       print(chunk, end="", flush=True)
    
    print()
    return chain

if __name__ == "__main__":
     demo_streaming()