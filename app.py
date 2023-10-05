import os

from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# os.environ.setdefault("OPENAI_API_KEY", "to be set")

app = Flask(__name__)

vectorstore = Chroma(persist_directory="./storage", embedding_function=OpenAIEmbeddings())

custom_template = """You are Tron Assistant. You are expert of Tron Chain, assisting devs on their doubts related to smart contract development on Tron as well as other chain specific queries. You have to strcitly answer only to questions related to Tron chain and not anything else.
Try to give brief explanation to the doubts asked and give code snippets wherever possible. I will give you context, if you want it then use it, otherwise reply according to your knowledge.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    model,
    vectorstore.as_retriever(),
    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    return_source_documents=True
)


@app.route("/chat", methods=["POST"])
def chat():
    """Chat with the Tron Assistant."""
    query = request.json["query"]
    chat_history = request.json.get("chat_history", [])
    # chat_history = [tuple(x) for x in chat_history]

    result = qa({"question": query, "chat_history": []})
    answer = result["answer"]
    source_documents = list(set([result["source_documents"][i].metadata["source"] for i in range(len(result["source_documents"]))]))

    response = {"answer": answer, "source_documents": source_documents}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
