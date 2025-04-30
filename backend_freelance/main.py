from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gleaming-youtiao-ad5a19.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Chain (memory assigned per request)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=retriever,
    memory=None,
    return_source_documents=False
)

# Request schema
class Question(BaseModel):
    question: str
    chat_history: list[str] = []

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(q: Question):
    promo = "\n\n🌱 Feeling stuck? Get a free fundraising consultation at www.nonprofitNavigator.pro"
    user_question = q.question.strip()

    # Chat history to LangChain memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    messages = []
    for i, msg in enumerate(q.chat_history):
        if i % 2 == 0:
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))
    memory.chat_memory.messages = messages

    # Debug: Print chat history
    print(f"\n📚 Chat history: {q.chat_history}")
    
    # Summary detection
    is_summary_request = "summarize" in user_question.lower() or "summary" in user_question.lower() or "summarise" in user_question.lower() or "recap" in user_question.lower() 
    if is_summary_request:
        print("📝 Detected summary request. Using GPT directly.")
        full_context = build_full_context(q.chat_history, user_question)
        gpt_answer = chat_llm.invoke(full_context).content
        return {"answer": gpt_answer + promo}

    # Check similarity scores
    docs_with_scores = vectorstore.similarity_search_with_score(user_question, k=3)
    if docs_with_scores:
        print(f"📄 Retrieved docs:")
        for doc, score in docs_with_scores:
            print(f"🔹 Score: {score:.4f} | Content: {doc.page_content[:100]}...")
    else:
        print("❌ No relevant documents found.")

    top_score = docs_with_scores[0][1] if docs_with_scores else 0
    threshold = 0.3
    print(f"🔎 Top similarity score: {top_score:.4f} | Threshold: {threshold}")

    if not docs_with_scores or all(score <= threshold for _, score in docs_with_scores):
        print("⚠️ No relevant context found. Using GPT only.")
        full_context = build_full_context(q.chat_history, user_question)
        answer = chat_llm.invoke(full_context).content
    else:
        print("✅ Relevant context found. Using retrieval-augmented generation.")
        qa_chain.memory = memory
        result = qa_chain.invoke({"question": user_question})
        answer = result["answer"]
        print(f"🤖 RAG Answer: {answer}")

        lowered = answer.lower()
        if (
            "i don't know" in lowered or
            "i am not sure" in lowered or
            ("don't" in lowered and "context" in lowered or
             "dont" in lowered and "information" in lowered)
        ):
            print("🔁 RAG answer was vague. Switching to GPT with full history.")
            full_context = build_full_context(q.chat_history, user_question)
            gpt_fallback = chat_llm.invoke(full_context).content

            if "don't" in lowered and "context" in lowered:
                gpt_fallback = "Can you please provide more details and context about it? :)\n\n" + gpt_fallback

            answer = gpt_fallback

    return {"answer": answer + promo}

def build_full_context(history: list[str], latest_question: str) -> str:
    """
    Format full chat history into a string and add the current question.
    """
    context = "Here's the chat so far:\n"
    for i, msg in enumerate(history):
        role = "User" if i % 2 == 0 else "Bot"
        context += f"{role}: {msg}\n"
    context += f"Now answer this: {latest_question}"
    return context

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
