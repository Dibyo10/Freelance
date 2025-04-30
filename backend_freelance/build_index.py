import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load files
pdf_files = glob.glob("data/*.pdf")
word_files = glob.glob("data/*.docx")

documents = []
for pdf in pdf_files:
    documents.extend(PyPDFLoader(pdf).load())
for docx in word_files:
    documents.extend(UnstructuredWordDocumentLoader(docx).load())

# Debug: Print number of loaded documents
print(f"Loaded {len(documents)} documents.")

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Debug: Print number of split documents
print(f"Split into {len(docs)} chunks.")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to local directory
vectorstore.save_local("faiss_index")
print("✅ FAISS index saved to 'faiss_index/'")
