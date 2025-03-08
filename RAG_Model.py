import streamlit as st
import os
import logging
import shutil
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama


logging.basicConfig(level=logging.INFO)


MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_pdf(doc_path):
    try:
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

@st.cache_resource
def load_vector_db(doc_path):
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    data = ingest_pdf(doc_path)
    if data is None:
        return None
    chunks = split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    vector_db.persist()
    logging.info("Vector database created and persisted.")
    return vector_db

def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate five alternative versions of the given user question 
        to retrieve relevant documents from a vector database. Provide these questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    template = """You are an AI assistant. If relevant context from the document exists, use it to answer.
    If no relevant document is found, answer using general knowledge.

    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created with hybrid answering logic.")
    return chain

def main():
    st.set_page_config(page_title="AI Document Assistant", layout="wide")
    st.markdown(
        """
        <style>
        .main { background-color: #f5f7fa; }
        .stTextInput>div>div>input { border-radius: 10px; }
        .stButton>button { border-radius: 8px; font-size: 16px; padding: 10px; }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("📄 AI Document Assistant")
    st.sidebar.title("ℹ️ About")
    st.sidebar.info("Upload a PDF and query it using AI.")
    uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
    if uploaded_file:
        doc_path = f"./data/{uploaded_file.name}"
        with open(doc_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ File uploaded successfully!")
        with st.spinner("🔄 Loading vector database..."):
            vector_db = load_vector_db(doc_path)
            if vector_db is None:
                st.error("❌ Failed to load or create the vector database.")
                return
        if st.button("🔄 Reset Database", use_container_width=True):
            shutil.rmtree(PERSIST_DIRECTORY)
            st.success("✅ Vector database reset successfully!")
        user_input = st.text_input("💬 Ask a question:", "")
        if user_input:
            with st.spinner("🤖 Thinking..."):
                try:
                    llm = ChatOllama(model=MODEL_NAME)
                    retriever = create_retriever(vector_db, llm)
                    retrieved_docs = retriever.invoke(user_input)
                    chain = create_chain(retriever, llm)
                    response = chain.invoke(input=user_input)
                    st.markdown("### 🤖 AI Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Enter a question to get started.")
    else:
        st.info("📂 Upload a PDF to begin.")

if __name__ == "__main__":
    main()
