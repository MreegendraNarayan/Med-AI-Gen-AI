from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import torch
import os

def download_hugging_face_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        raise