from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import os

def demo_init_chat_model():
    #Demo generic init_chat_model fucntion for all model providers
    chat_model = init_chat_model(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    streaming=True,
    #api_key=os.environ["GROQ_API_KEY"],
    model_provider="groq",
    max_retries=3
    )

    response = chat_model.invoke("what is the capital of France? Answer in one word.")
    print(f"Response: {response.content}")

    if os.getenv("GROQ_API_KEY"):
        chat_model = init_chat_model(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        streaming=True,
        model_provider="groq",
        max_retries=3
        )

    response = chat_model.invoke("what is the capital of India? Answer in one word.")
    print(f"Response from Second Model: {response.content}")
    return chat_model

def demo_model_comparison():
    # demo comparing different model outputs using loop
    prompt = "Explain recursion in one sentence."
    models = {
    "gpt-oss": init_chat_model(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    streaming=True,
    #api_key=os.environ["GROQ_API_KEY"],
    model_provider="groq",
    max_retries=3
    ),
    "qwen": init_chat_model(
    model="qwen/qwen3.8-27b",
    temperature=0.7,
    streaming=True,
    #api_key=os.environ["GROQ_API_KEY"],
    model_provider="groq",
    max_retries=3
    )
    }

    print(f"Prompt:{prompt}")
    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"response from {model_name} is {response.content}.\n")

def demo_message():

    #demo system and human message
    model =  init_chat_model(
    model="openai/gpt-oss-120b",
    temperature=0,
    streaming=True,
    #api_key=os.environ["GROQ_API_KEY"],
    model_provider="groq",
    max_retries=3
    )

    messages = [
        SystemMessage(content="You are a helpful physician."),
        HumanMessage(content="What is the cure for cough and cold. Answer in one line.")
    ]

    response = model.invoke(messages)
    print(f"Response using message object:{response.content}")

    messages.append(response)
    print(f"Appended Messages:{messages}")

    messages.append(HumanMessage("What about fever?"))

    print("\nMulti-turn conversation")
    response = model.invoke(messages)
    print(f"Follow up response: {response}")

if __name__ == "__main__":
     demo_init_chat_model()
     demo_model_comparison()
     demo_message()