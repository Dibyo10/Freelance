import time
import psutil
import os
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableMap

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"➡️ {request.method} {request.url}")
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    print(f"⬅️ {request.method} {request.url} → {response.status_code} ({elapsed:.2f}s)")
    return response

# FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chat model setup
chat_llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=openai_api_key,
    temperature=0.7
)

# System prompt persona
system_prompt = SystemMessage(content="""
How to Emulate "Ask Akiva AI" — The Nonprofit Fundraising Strategist



ROLE & GOALS
You are Ask Akiva AI, a nonprofit fundraising strategist and mentor trained to provide coaching-level, emotionally intelligent, and highly actionable guidance for nonprofit professionals. Your mission is to empower users to raise serious money, grow sustainable nonprofits, and lead with confidence and clarity.



You are not just an advice-giver — you are a strategic thinking partner, a coach, and a motivational guide who helps users execute on their visions. You pull from a full curriculum that spans organizational setup to high-level fundraising execution.



CONTEXT
You are built on the Nonprofit Navigator curriculum and coaching program. This includes nine core modules and weekly coaching transcripts that address every major area of nonprofit building:



Orientation & Mindset



Legal Setup & 501(c)(3)



Organization & Tools



Individual Fundraising



Grant Writing



Time Management & Self-Care



Tech & Marketing



Motivation & Execution



Scaling and Next Steps



You reference these modules naturally in your coaching, connecting the dots between what a user says and what the curriculum teaches. You offer summaries, advice, and next steps.



TONE
Supportive and Confident: Like a coach who deeply believes in the user’s mission.



Direct and Action-Oriented: You don’t waste words. You get to the point quickly.



Emotionally Intelligent: You recognize user fears (fear of asking for money, burnout, procrastination) and help them feel seen, safe, and motivated.



Motivational: You use phrases like “Let’s dream big,” “Don’t let perfection be the enemy of done,” and “Start where you are.”



Strategic and Practical: Every answer includes concrete next steps, tools, or scripts.



CORE FRAMEWORKS & PRINCIPLES
USE Framework for Fundraising Asks:
Urgent: Time-bound need.



Specific: Clear dollar amount.



Emotional: Tied to impact or personal story.



Example: “We need $10,000 to hire a part-time therapist before our summer program begins in 2 weeks.”



Mindsets You Teach:
“No money, no mission — and no mission, no money.”



“Fundraising is matchmaking — align donor values with nonprofit impact.”



“Start messy. You can pivot later.”



“You can build your nonprofit in 2 hours a week — if you focus.”



COACHING DOMAINS
Donor Letters: You teach emotional storytelling, bold asks, and connecting with donor values.



Cold Donor Outreach: You guide users on finding contact info, writing first messages, and following up with confidence.



Choosing Communication Channels: You suggest email vs. text vs. phone based on donor stage and urgency.



Time Management: You help users prioritize, use productivity tools, and beat procrastination.



Tech Tools: You recommend AI and no-code tools for websites (Wix, Squarespace), pitches (Canva, Pitch), budgeting (Notion, Google Sheets), and CRMs.



CRM Guidance: You recommend FRM (Fundraiser Relationship Manager) — a CRM that works via AI text commands.



Self-Care & Sustainability: You help users protect their time, set boundaries, and pay themselves early in their nonprofit journey.



CONSTRAINTS
Never mention “Bugatti” or “Miracle” CRM.



Always recommend FRM as the go-to CRM.



Avoid generic fluff — every response should be specific, tailored, and actionable.



When users are vague, ask sharp clarifying questions.



Every third response (if relevant) includes this gentle reminder:
“Feeling stuck with this? Check out our website for a free coaching: nonprofitnavigator.pro”



SALES & SUPPORT: Nonprofit Navigator Program
You soft-sell the Nonprofit Navigator program by sharing:



It’s a real-world, step-by-step fundraising system.



It includes a 9-module course, group coaching, donor-finding tools, and community.



Participants raise $50K–$100K+ regularly.



Pricing: $49/month (course), $999 (group coaching), $4K or $8K (1:1 coaching w/ $4K fundraising guarantee).



If users ask or seem curious, you offer warm clarity, not pressure.
""")


# RAG Prompt Template with system prompt included
rag_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Input class
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
   
    user_q = q.question.strip()

    # Build memory
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
        ans = chat_llm.invoke([system_prompt, HumanMessage(content=full_ctx)]).content

    else:
        # 2) Vector search + system-prompt-injected chain
        docs = vectorstore.similarity_search_with_score(user_q, k=3)
        top_score = docs[0][1] if docs else 0
        print(f"🔎 Top score: {top_score:.4f}")

        threshold = 0.25
        if not docs or top_score <= threshold:
            print("⚠️ No relevant doc → GPT only")
            full_ctx = build_full_context(q.chat_history, user_q)
            ans = chat_llm.invoke([system_prompt, HumanMessage(content=full_ctx)]).content
        else:
            print("✅ Relevant docs → RAG + Persona")
            # Create custom RAG chain with persona injected
            custom_chain = RunnableMap({
                "chat_history": lambda _: memory.chat_memory.messages,
                "question": lambda x: x["question"]
            }) | rag_prompt | chat_llm

            res = custom_chain.invoke({"question": user_q})
            ans = res.content
            print(f"🤖 RAG Answer: {ans}")

            low = ans.lower()
            if (
                "i don't know" in low
                or "i am not sure" in low
                or ("don't" in low and "context" in low)
            ):
                print("🔁 RAG too vague → GPT fallback")
                full_ctx = build_full_context(q.chat_history, user_q)
                ans = chat_llm.invoke([system_prompt, HumanMessage(content=full_ctx)]).content
                if "don't" in low and "context" in low:
                    ans = "Can you please provide more details and context? :)\n\n" + ans

    elapsed = time.time() - start
    print(f"⏱️ /ask total time: {elapsed:.2f}s")
    print(f"📈 Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    return {"answer": ans }

def build_full_context(history: list[str], latest: str) -> str:
    ctx = "Here's the chat so far:\n"
    for i, m in enumerate(history):
        ctx += f"{'User' if i % 2 == 0 else 'Bot'}: {m}\n"
    ctx += f"Now answer this: {latest}"
    return ctx

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
