import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

        .stApp {
            background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(10, 10, 15) 90%);
            font-family: 'Inter', sans-serif;
        }

        section[data-testid="stSidebar"] {
            background-color: #0F1116;
            border-right: 1px solid #1E293B;
        }
        
        div[data-testid="stFileUploader"] {
            border: 1px dashed #007794;
            border-radius: 10px;
            background-color: rgba(0, 0, 148, 0.05);
            transition: all 0.3s ease;
        }
        div[data-testid="stFileUploader"]:hover {
            background-color: rgba(0, 0, 148, 0.1);
            box-shadow: 0 0 15px rgba(0, 0, 148, 0.3);
        }

        div.stButton > button {
            background: linear-gradient(45deg, #00A6FF, #32DDFF);
            color: #000;
            font-family: 'JetBrains Mono', monospace;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.1s;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px #64DDFF;
        }
        div.stButton > button:active {
            transform: scale(0.98);
        }

        div[data-testid="stChatMessage"]:nth-child(odd) {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 15px 15px 0 15px;
        }
        
        div[data-testid="stChatMessage"]:nth-child(even) {
            background: rgba(16, 185, 129, 0.05);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 15px 15px 15px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
        }

        h1, h2, h3 {
            color: #E2E8F0;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -1px;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0F1116; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }

    </style>
    """, unsafe_allow_html=True)


st. set_page_config("Fact-Checker", layout='centered')
inject_custom_css()
st.title("Fact-Checking Agent")

PINECONE_SPI_KEY = st.secrets["pinecone_api_key"]
GROQ_API_KEY = st.secrets["groq_api_key"]
INDEX_NAME = "rag"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

try:
    pc = Pinecone(PINECONE_SPI_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Failed to connect to Pinecone: {e}")
    
def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.read())
        path = temp_file.name
    
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs
    finally:
        if os.path.exists(path):
            os.unlink(path)

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Read PDF"):
        with st.spinner("Reading your PDF..."):
            
            try:
                pc = Pinecone(api_key=PINECONE_SPI_KEY)
                index = pc.Index(INDEX_NAME)
                index.delete(delete_all=True)
            except Exception as e:
                st.warning(f"Could not clear index: {e}")
            
            st.session_state.messages = []
            
            docs = process_file(uploaded_file)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            try:
                vectorstore = PineconeVectorStore.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    index_name=INDEX_NAME
                )
                st.success(f"Read {len(splits)} chunks")
            except Exception as e:
                st.error(f"Error with Pinecone: {e}")

    
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name='llama-3.1-8b-instant',
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    If the answer is not in the context, say "I cannot find this information in the documents."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if prompt_input := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": prompt_input})
            answer = response['answer']
            st.markdown(answer)
            
            with st.expander("View Sources"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"Source {i + 1}: {doc.page_content[:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})