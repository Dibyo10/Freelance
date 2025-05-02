# main.py
import time
import psutil
import os
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

app = FastAPI()

# 1) Mount CORS first so OPTIONS never 400
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://askakivai.vercel.app","https://www.askakivai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Simple request/response logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"➡️ {request.method} {request.url}")
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    print(f"⬅️ {request.method} {request.url} → {response.status_code} ({elapsed:.2f}s)")
    return response

# Embeddings + FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# RAG chain (memory will be set per request)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=retriever,
    memory=None,
    return_source_documents=False
)

class Question(BaseModel):
    question: str
    chat_history: list[str] = []

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def health_check():
    return {"status": "ok"}

@app.get("/memory")
def memory_usage():
    proc = psutil.Process()
    m = proc.memory_info()
    return {
        "rss_MB": round(m.rss / 1024 / 1024, 2),
        "vms_MB": round(m.vms / 1024 / 1024, 2),
    }

@app.post("/ask")
async def ask_question(q: Question):
    start = time.time()
    promo = "\n\n🌱 Feeling stuck? Get a free fundraising consultation at www.nonprofitNavigator.pro"
    user_q = q.question.strip()

    # Build LangChain memory from chat_history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    msgs = []
    for i, m in enumerate(q.chat_history):
        msgs.append(HumanMessage(content=m) if i % 2 == 0 else AIMessage(content=m))
    memory.chat_memory.messages = msgs

    print(f"\n📚 Chat history: {q.chat_history}")

    # 1) Summary/directive bypass
    if any(kw in user_q.lower() for kw in ["summarize", "summary", "summarise", "recap"]):
        print("📝 Summary requested → GPT only")
        full_ctx = build_full_context(q.chat_history, user_q)
        ans = chat_llm.invoke(full_ctx).content
    else:
        # 2) Vector search + threshold
        docs = vectorstore.similarity_search_with_score(user_q, k=3)
        top_score = docs[0][1] if docs else 0
        print(f"🔎 Top score: {top_score:.4f}")

        threshold = 0.30
        if not docs or top_score <= threshold:
            print("⚠️ No relevant doc → GPT only")
            full_ctx = build_full_context(q.chat_history, user_q)
            ans = chat_llm.invoke(full_ctx).content
        else:
            print("✅ Relevant docs → RAG")
            qa_chain.memory = memory
            res = qa_chain.invoke({"question": user_q})
            ans = res["answer"]
            print(f"🤖 RAG Answer: {ans}")

            low = ans.lower()
            if (
                "i don't know" in low
                or "i am not sure" in low
                or ("don't" in low and "context" in low)
            ):
                print("🔁 RAG too vague → GPT fallback")
                full_ctx = build_full_context(q.chat_history, user_q)
                ans = chat_llm.invoke(full_ctx).content
                if "don't" in low and "context" in low:
                    ans = "Can you please provide more details and context? :)\n\n" + ans

    elapsed = time.time() - start
    print(f"⏱️ /ask total time: {elapsed:.2f}s")
    print(f"📈 Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    return {"answer": ans + promo}


def build_full_context(history: list[str], latest: str) -> str:
    ctx = "Here's the chat so far:\n"
    for i, m in enumerate(history):
        ctx += f"{'User' if i % 2 == 0 else 'Bot'}: {m}\n"
    ctx += f"Now answer this: {latest}"
    return ctx


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
