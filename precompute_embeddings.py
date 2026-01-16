# precompute_embeddings.py
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load your trained model
model = SentenceTransformer("legal_nlp_model")

# Load your categorized dataset
df = pd.read_csv("Final_QA_dataset.csv")
questions = df['Question'].tolist()

# Encode all questions
corpus_embeddings = model.encode(questions, convert_to_tensor=True)

# Save embeddings
torch.save(corpus_embeddings, "corpus_embeddings.pt")
print("✅ Saved corpus_embeddings.pt successfully.")


