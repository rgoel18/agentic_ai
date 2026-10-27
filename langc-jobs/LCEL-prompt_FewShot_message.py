from dotenv import load_dotenv
load_dotenv()

import sys
from importlib import metadata  # Safe way to fetch installed package versions
from langchain_core import __version__ as  core_version
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, ChatMessage

prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about a {topic}")

messages1 = prompt.format_messages(adjective="funny", topic="princess")

# multi-message template
prompt = ChatPromptTemplate.from_messages(
                [ 
                    (
                        "system",
                        "You are a helpful assistant that translates {input_language} to {output_language}"
                    ),
                    (
                        "human",
                        "Translate the following {text}"
                    )                      
                ]
)

messages2 = prompt.format_messages(
    input_language = "English",
    output_language = "French",
    text = " I love my programming"
)

chat_model = init_chat_model(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            streaming=True,
            #api_key=os.environ["GROQ_API_KEY"],
            model_provider="groq",
            max_retries=3
            )

response = chat_model.invoke(messages1)
print(response.content)

response = chat_model.invoke(messages2)
print(response.content)

messages = [
    HumanMessage(content = "Hello"),
    AIMessage(content = "Hi there! How can I assist you today?"),
    SystemMessage(content = "This is a system message."),
    #ToolMessage(content = "This is a tool message."),
    #ChatMessage(content = "This is a Chat Message.")
]

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input":"happy","output":"sad"},
    {"input":"tall", "output":"short"}
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="{input}"), 
        AIMessage(content="{output}")
        ]
    )

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = examples
)

final_prompt = ChatPromptTemplate.from_messages(
[
    ("system", "give the opposite of each word"),
    fewshot_prompt,
    ("human","{input}")
]
)

chat_model = init_chat_model(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            streaming=True,
            #api_key=os.environ["GROQ_API_KEY"],
            model_provider="groq",
            max_retries=3
            )

response = chat_model.invoke(final_prompt.format_messages(input = "fat"))
print(f"Fewshot response: {response}")