from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS vector store
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Setup QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=True
)

# Pydantic model for request body
class Question(BaseModel):
    question: str

# Root route for Render health check
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def read_root():
    return {"status": "ok"}

# Main /ask endpoint with fallback
@app.post("/ask")
async def ask_question(q: Question):
    response = qa.invoke(q.question)
    base_answer = response["result"]
    source_documents = response.get("source_documents", [])

    promo = "\n\n🌱 Feeling stuck? Get a free fundraising consultation at www.nonprofitNavigator.pro"

    # If no good documents are retrieved, fallback to general GPT response
    if not source_documents or all(not doc.page_content.strip() for doc in source_documents):
        print("🔄 No relevant docs found. Falling back to general GPT answer.")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
        base_answer = llm.invoke(q.question)

    return {"answer": base_answer + promo}

# Local dev entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)