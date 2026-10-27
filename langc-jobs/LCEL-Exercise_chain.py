from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def demo_basic_chain():
    '''demonstrate a basic chain using LCEL and runnables'''
    #Component1:Define the prompt template using a LCEL
    prompt = ChatPromptTemplate.from_template("Create a one liner marketing tagline for product {product} considering the audience as {audience}.")
    model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
    parser = StrOutputParser()

    # Compose with Pipe operator
    chain = prompt|model|parser 

    # Execute chain with an input
    result = chain.invoke({"product":"Shampoo", "audience":"kids"})
    print(f"Response:{result}")

    return chain

#def demo_batch_execution():


if __name__ == "__main__":
    demo_basic_chain()

