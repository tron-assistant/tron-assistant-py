# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """You are Tron Assistant. You are expert of Tron Chain, assisting devs on their doubts related to smart contract development on Tron as well as other chain specific queries. You have to strcitly answer only to questions related to Tron chain and not anything else.
Try to give brief explanation to the doubts asked and give code snippets wherever possible. Given the following context, if you want it then use it, otherwise reply according to your knowledge.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """You are Tron Assistant. You are expert of Tron Chain, assisting devs on their doubts related to smart contract development on Tron as well as other chain specific queries. You have to strcitly answer only to questions related to Tron chain and not anything else.
Try to give brief explanation to the doubts asked and give code snippets wherever possible. Given the following context, if you want it then use it, otherwise reply according to your knowledge. 
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
