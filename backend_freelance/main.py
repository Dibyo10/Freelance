from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
import uvicorn
import glob
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# ...
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load prebuilt FAISS index from disk
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    response = qa.invoke(q.question)
    return {"answer": response["result"]}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)
