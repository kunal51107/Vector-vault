#  Vector-Vault

![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)
![Python](https://img.shields.io/badge/Python-3.13-blue)

### A Vector Database Built from Scratch 

> ** PROJECT UNDER ACTIVE DEVELOPMENT **
> *I am building this to understand the internal mechanics of Vector Search Engines like Pinecone/ChromaDB.*

---

##  The Goal
Most people use `pip install chromadb`. I want to build the engine itself.
This project implements:
1.  **Vector Mathematics** (Cosine Similarity, Euclidean Distance) manually.
2.  **Indexing Algorithms** (Inverted File Index / K-Means) for speed.
3.  **Quantization** (Compression) for memory optimization.
4.  **TCP Server** to handle client requests.

---

## The Roadmap

- [x] **Phase 1: Project Setup** (Folder structure, Environment, CI/CD)
- [x] **Phase 2: The Math Core** (Numpy Vector Logic, Unit Tests)
- [ ] **Phase 3: The Flat Index** (Brute force search implementation)
- [ ] **Phase 4: Optimization** (IVF Indexing & Quantization)
- [ ] **Phase 5: The Server** (Socket programming & API)

---

## Tech Stack
* **Language:** Python (Strict Typing)
* **Math:** NumPy (Linear Algebra)
* **Testing:** Pytest

---

*Created by Kunal*