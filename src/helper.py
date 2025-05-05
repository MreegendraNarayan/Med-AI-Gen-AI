from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

def download_hugging_face_embeddings():
    # Forcing CPU usage and optimizing memory
    model_kwargs = {
        'device': 'cpu',
        'torch_dtype': torch.float32,
        'max_length': 256  # Limiting sequence length
    }
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings