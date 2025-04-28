from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
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

# Smarter retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Setup memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Chat model
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Setup Conversational QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# Pydantic model for request body
class Question(BaseModel):
    question: str

# Root route for Render health check
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def read_root():
    return {"status": "ok"}

# Main /ask endpoint
@app.post("/ask")
async def ask_question(q: Question):
    promo = "\n\n🌱 Feeling stuck? Get a free fundraising consultation at www.nonprofitNavigator.pro"

    user_question = q.question.strip()

    # --- 🧠 Short/generic question detection ---
    if len(user_question.split()) <= 3:
        print("🔎 Question too short/generic. Skipping retrieval. Using GPT only.")
        answer = chat_llm.invoke(user_question)
    else:
        print("✅ Good question. Using Retrieval with memory.")
        result = qa.invoke({"question": user_question})
        answer = result["answer"]

    return {"answer": answer + promo}

# Local dev entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
