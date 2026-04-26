# 🛠️ TechNova Support RAG Assistant

An advanced **Agentic Retrieval-Augmented Generation (RAG)** system designed to provide technical support for TechNova X1000 routers. This project uses **LangGraph** for sophisticated workflow orchestration and includes a **Human-In-The-Loop (HITL)** escalation mechanism.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=googlecloud&logoColor=white)

---

## 🌟 Key Features

*   **Agentic Workflow**: Uses LangGraph to classify user intent and grade document relevance before answering.
*   **Corrective RAG**: If retrieved documents are irrelevant, the system identifies the gap and escalates instead of hallucinating.
*   **Human-In-The-Loop (HITL)**: Integrated hand-off panel for human agents to take over complex or sensitive queries.
*   **Context-Aware**: Maintains conversation history using `MemorySaver` checkpointing.
*   **High Precision**: Uses `all-MiniLM-L6-v2` embeddings for accurate technical retrieval.

---

## 🏗️ Architecture

The system follows a non-linear graph-based path:
1.  **Classify**: Is this a technical or account-related question?
2.  **Retrieve**: Fetch relevant chunks from the manual.
3.  **Grade**: Are these chunks actually useful?
4.  **Generate/Escalate**: Either provide a grounded answer or ask a human for help.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/TechNova-RAG-Assistant.git
cd TechNova-RAG-Assistant
```

### 2. Set Up Environment
Create a `.env` file in the root directory:
```text
HF_TOKEN=your_huggingface_access_token
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest the Knowledge Base
Run this script to process the PDF and populate the vector store:
```bash
python ingestion.py
```

### 5. Launch the Assistant
```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack

*   **LLM**: Zephyr-7B-Beta (via HuggingFace Endpoint)
*   **Orchestration**: LangGraph, LangChain
*   **Vector DB**: ChromaDB
*   **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
*   **UI**: Streamlit

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
Distributed under the MIT License.
