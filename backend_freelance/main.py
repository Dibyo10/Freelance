from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
import glob
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings with the correct API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load all PDF and Word files from the data folder
pdf_files = glob.glob("data/*.pdf")
word_files = glob.glob("data/*.docx")

documents = []

# Load PDFs
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

# Load Word docs
for docx in word_files:
    loader = UnstructuredWordDocumentLoader(docx)
    documents.extend(loader.load())

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Create vector store with OpenAI embeddings
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

# Request body format
class Question(BaseModel):
    question: str

# API endpoint to answer questions
@app.post("/ask")
async def ask_question(q: Question):
    response = qa.invoke(q.question)
    return {"answer": response["result"]}
