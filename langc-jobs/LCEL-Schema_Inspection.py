from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def demo_schema_inspection():
    #Demonstrate streaming for real-time output.
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    #reasoning_effort="low",
    #max_tokens=8000, 
    )
    parser = StrOutputParser()

    chain = prompt | model | parser
    #Inspect input and output schema
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")
    return chain

if __name__ == "__main__":
     demo_schema_inspection()