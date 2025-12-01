import numpy as np
from typing import Union

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length
    v_norm = v / ||v||
    
    Why? Cosine Similarity is just Dot Product if vectors are of magnitude 1.
    This makes search 2x faster later.
    """
    # Handle 1D vector (Single Query)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    
    # Handle 2D matrix (The Database)
    # axis=1 means calculate magnitude for each row separately
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Avoid division by zero (replace 0 with 1 in the divisor)
    norms = np.where(norms == 0, 1, norms)
    
    return vectors / norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute Cosine Similarity.
    Range: 1.0 (Identical) to -1.0 (Opposite).
    """
    # 1. Normalize both inputs
    a_norm = normalize_vectors(a)
    b_norm = normalize_vectors(b)
    
    # 2. Compute Dot Product
    # Case 1: Both are single vectors
    if a_norm.ndim == 1 and b_norm.ndim == 1:
        return np.dot(a_norm, b_norm)
    
    # Case 2: One is a database (Matrix), one is a query (Vector)
    # This is "Broadcasting" - checking one query against 1 Million rows instantly
    elif a_norm.ndim == 2 and b_norm.ndim == 1:
        return np.dot(a_norm, b_norm)
    
    # Case 3: Matrix vs Matrix (Rare)
    else:
        return np.dot(a_norm, b_norm.T)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute L2 (Euclidean) Distance.
    Lower is better. 0.0 means identical.
    """
    if a.ndim == 1 and b.ndim == 1:
        return np.linalg.norm(a - b)
    else:
        # Broadcasting logic for Matrix - Vector
        return np.linalg.norm(a - b, axis=1)