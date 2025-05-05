from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

def download_hugging_face_embeddings():
    model_kwargs = {
        'device': 'cpu',
        'max_length': 256,
        'model_max_length': 256
    }

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        raise