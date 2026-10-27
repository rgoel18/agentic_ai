from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def demo_batch_execution():
    "Demonstrate a basic batch chain using LCEL and runnables."
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    )
    parser = StrOutputParser()

    chain = prompt | model | parser
    inputs = [{"text":"Hello, how are you?"}, {"text":"What is your name?"}, {"text":"What is the nearest retaurant?"}]
    response = chain.batch(inputs)

    for out in zip(inputs,response):
        print(f"Input:{out[0]['text']} => Output:{out[1]}")
    
    return chain

if __name__ == "__main__":
     demo_batch_execution()