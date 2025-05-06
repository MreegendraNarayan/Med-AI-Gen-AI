from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Missing required API keys in .env file")

from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("Installing langchain-google-genai...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-google-genai==0.0.6"])
    from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("user-input", "")

        if not msg:
            return "Please ask a medical question."

        print(f"Received message: {msg}")


        response = rag_chain.invoke({"input": msg})


        answer = response.get("answer", "I'm sorry, I couldn't process that request.")
        print(f"Response: {answer}")

        return answer

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"error": str(error)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8080))
    
    # In development, you might want debug=True
    if os.environ.get("FLASK_ENV") == "development":
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        # Production mode
        app.run(host="0.0.0.0", port=port, debug=False)