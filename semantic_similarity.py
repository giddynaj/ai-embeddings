import numpy as np
from embeddings import embed

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = [
    "The dog chased the ball",
    "A puppy ran after a sphere",
    "The stock market crashed",
]

embeddings = [ embed(s) for s in sentences]

for i in range(1, len(sentences)):
    score = cosine_similarity(embeddings[0], embeddings[i])
    print(f"Similarity to '{sentences[i]}': {score:.4f}")
