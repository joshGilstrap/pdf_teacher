# The Fact-Checker: Citation-Aware RAG Agent

An Enterprise-Grade RAG (Retrieval Augmented Generation) system that answers questions based strictly on uploaded documents, providing granular citations to prevent hallucinations.

## Live Demo

[Direct link](https://pdf-fact-checker.streamlit.app/) or use it directly on my [portfolio website](joshgilstrap.com)

<img width="864" height="815" alt="Screenshot 2025-11-27 200319" src="https://github.com/user-attachments/assets/cd52a713-97e4-454e-a099-54f3119415c7" />

## The Problem & Solution

The Problem: Most "Chat with PDF" tutorials suffer from "Hallucination" (making things up) and "Context Bleeding" (mixing up data from old documents).
The Solution: This application implements a Strict Citation Architecture.

1. It uses a Recursive Character Splitter to maintain semantic context.

2. It forces the LLM (Llama 3.1) to cite its sources or admit ignorance.

3. It implements a "Wipe-and-Replace" indexing strategy to ensure data hygiene between sessions.

## Architecture Pipeline

1. Ingestion: User uploads a PDF.

2. Processing:

  - Loader: ```PyPDFLoader``` extracts text via a tempfile buffer (OS-safe).

  - Chunking: ```RecursiveCharacterTextSplitter``` (Size: 1000, Overlap: 200) creates overlapping semantic tiles.

3. Embedding: ```HuggingFaceEmbeddings``` (```all-MiniLM-L6-v2```) converts text to vectors locally (CPU-optimized).

4. Storage: Vectors are upserted to a Pinecone Serverless index.

5. Retrieval: Cosine Similarity search fetches top-k relevant chunks.

6. Generation: Groq generates an answer using only the retrieved context.

## Tech Stack & Engineering Decisions

| Component | Technology | Reasoning |
| :-------- | :--------- | :-------- |
| Vector DB | Pinecone Serverless | Scalable, managed vector storage with low latency and a generous free tier. |
| Inference | Groq (Llama 3.1) | Ultra-low latency (~300 T/s) makes the RAG experience feel "real-time" compared to GPT-4. |
| Embeddings | HuggingFace | ```all-MiniLM-L6-v2``` is free, fast, and runs locally within the Streamlit container, saving API costs. |
| Frontend | Streamlit | Built-in chat interface (```st.chat_message```) allows for rapid UI iteration. |

## Key Features

1. Zero-Hallucination Prompting

The prompt template is engineered to restrict the model's creativity:

"Answer the questions based on the provided context only... If the answer is not in the context, say 'I cannot find this information'."

2. Transparent Citations

Unlike black-box answers, this agent includes a "View Sources" expander. It reveals the exact raw text chunks retrieved from Pinecone used to generate the answer, building user trust.

3. State Management Strategy

To prevent "Context Bleeding" (where Document A answers questions about Document B):

  - I implemented a ```index.delete(delete_all=True)``` trigger on every new file upload.

  - This ensures the Vector Database always reflects the current document state 1:1.

## Local Installation

Prerequisites: Python 3.10+, Pinecone API Key, Groq API Key.

Clone the repository
```
git clone [https://github.com/joshgilstrap/fact-checker.git](https://github.com/joshgilstrap/fact-checker.git)
cd fact-checker
```

Install dependencies
```
pip install -r requirements.txt
```

Configure Secrets
Create a folder .streamlit and a file secrets.toml:
```
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
PINECONE_API_KEY = "pcsk_..."
```

Run the App
```
streamlit run rag_app.py
```

## Key Code Snippet: The Ingestion Logic
```
# Processing the PDF with a safe tempfile handler
def process_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        return docs
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
```

## Future Improvements

[ ] Hybrid Search: Implement BM25 + Vector Search for better keyword matching.

[ ] Multi-Document Support: Use Pinecone Namespaces to allow querying across multiple PDFs simultaneously.

[ ] Memory: Add ```ConversationBufferMemory``` to allow follow-up questions (currently single-turn RAG).

## Connect

Built by Josh Gilstrap as a showcase of Production RAG Systems.

[LinkedIn](https://www.linkedin.com/in/josh-gilstrap-3b34b0126/)

[Portfolio](joshgilstrap.com)
