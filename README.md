Maargi AI is an insider, AI-driven travel guide and discovery platform tailored for exploring offbeat trails, hidden homestays, and undiscovered locations in Uttarakhand, India. 

This repository contains the backend core engine, orchestrating open-source Large Language Models (LLMs) and local embedding models for low-latency, cost-effective travel recommendation systems.

---

## 🛠️ Tech Stack & Architecture

- **LLM Orchestration**: [LangChain](https://python.langchain.com/) (`langchain-huggingface`, `langchain-core`)
- **Language Model**: Meta **Llama 3.1 (8B Instruct)** via Hugging Face Serverless Inference API
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, running locally on CPU)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) *(Integration Ready)*
- **Configuration & Secrets**: `python-dotenv`
- **Language & Runtime**: Python 3.10+

---

## 📁 Project Directory Structure

```text
Maargi.AI/
├── app/
│   └── services/
│       └── agent.py          # Core LangChain agent with Llama 3.1 & memory setup
├── .env                      # Local secret keys & API tokens (Git-ignored)
├── .env.example              # Template for environment configuration
├── .gitignore                # Rules for excluding secrets, venv, and build files
└── README.md                 # Project documentation
