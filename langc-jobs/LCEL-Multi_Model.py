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

def demo_multi_model():
    #Demo func take question and list of models and return dict of response

    def get_response(question: str, model_names = list[str]) -> dict:
        responses = {}
        for model_name in model_names:
            chat_model = init_chat_model(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            streaming=True,
            #api_key=os.environ["GROQ_API_KEY"],
            model_provider="groq",
            max_retries=3
            )

            response = chat_model.invoke(question)
            responses[model_name]=response.content
        return responses

    results = get_response("what is AI? Answer in 2 lines.",["openai/gpt-oss-120b","qwen/qwen3.8-27b"])
    for model, answer in results.items():
        print(f"Response from model {model}: {answer}")

if __name__ == "__main__":
     demo_multi_model()