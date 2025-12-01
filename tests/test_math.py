import numpy as np
import pytest
from vector_vault.utils.math import normalize_vectors, cosine_similarity , euclidean_distance

def test_normalization():
    # A vector [3, 4] has length 5. Normalized, it should have length 1.
    vec = np.array([3.0, 4.0])
    norm_vec = normalize_vectors(vec)
    
    # Check if length is 1.0
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)
    # Check values (3/5 = 0.6, 4/5 = 0.8)
    assert np.allclose(norm_vec, np.array([0.6, 0.8]))

def test_cosine_identical():
    # Identical vectors must have similarity 1.0
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])
    score = cosine_similarity(v1, v2)
    assert np.isclose(score, 1.0)

def test_cosine_orthogonal():
    # Perpendicular vectors (90 degrees) must have similarity 0.0
    v1 = np.array([1.0, 0.0]) # X axis
    v2 = np.array([0.0, 1.0]) # Y axis
    score = cosine_similarity(v1, v2)
    assert np.isclose(score, 0.0)

def test_matrix_search():
    # Simulate a Database of 3 vectors
    db = np.array([
        [0.9, 0.1], # Vector A (Target)
        [0.1, 0.9], # Vector B
        [-0.9, -0.1] # Vector C (Opposite)
    ])
    
    query = np.array([0.9, 0.1]) # Searching for Vector A
    
    scores = cosine_similarity(db, query)
    
    # The first score should be highest (1.0)
    assert np.argmax(scores) == 0
    assert scores[0] > scores[1]

def test_euclidean_batch_search():
    # Database: 2 Vectors (2D)
    # Vec A is at origin (0,0)
    # Vec B is far away (3,4)
    db = np.array([
        [0.0, 0.0],
        [3.0, 4.0]
    ])
    
    # Query Batch: 2 Queries (2D)
    # Q1 looks for Vec A
    # Q2 looks for Vec B
    queries = np.array([
        [0.0, 0.0], 
        [3.0, 4.0]
    ])
    
    # The result should be a 2x2 Matrix
    # [ Dist(Q1, A), Dist(Q1, B) ]
    # [ Dist(Q2, A), Dist(Q2, B) ]
    distances = euclidean_distance(db, queries)
    
    # Check Shape: (2 DB vectors, 2 Queries) -> (2, 2)
    assert distances.shape == (2, 2)
    
    # Check Logic:
    # Q1 (0,0) vs A (0,0) -> Distance 0
    assert distances[0, 0] == 0.0
    
    # Q2 (3,4) vs B (3,4) -> Distance 0
    assert distances[1, 1] == 0.0
    
    # Q1 (0,0) vs B (3,4) -> Distance 5 (3-4-5 triangle)
    assert distances[0, 1] == 5.0