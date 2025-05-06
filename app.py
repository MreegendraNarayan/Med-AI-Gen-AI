from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import gc
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Missing required API keys in .env file")

from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = None
llm = None
docsearch = None
rag_chain = None

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def initialize_components():
    global embeddings, llm, docsearch, rag_chain
    try:
        if embeddings is None:
            embeddings = download_hugging_face_embeddings()
            clear_memory()

        if llm is None:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.4,
                max_output_tokens=500,
                google_api_key=GOOGLE_API_KEY
            )
            clear_memory()

        if docsearch is None:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name="medicalbot",
                embedding=embeddings
            )
            clear_memory()

        if rag_chain is None:
            retriever = docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            clear_memory()

    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/get", methods=["POST"])
def chat():
    try:
        if rag_chain is None:
            initialize_components()

        msg = request.form.get("user-input", "")
        if not msg:
            return "Please ask a medical question."

        logger.info(f"Processing message: {msg}")
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm sorry, I couldn't process that request.")

        clear_memory()
        return answer

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

@app.after_request
def cleanup(response):
    clear_memory()
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    initialize_components()
    app.run(host="0.0.0.0", port=port)