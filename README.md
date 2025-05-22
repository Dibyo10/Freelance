# 🧠 Custom-Trained AI Chatbot for Fundraising Docs

This is a **custom-trained AI chatbot** built for a freelance client to help answer fundraising-related questions — strictly based on their internal documents (PDFs, Word files).

The project combines **LangChain**, **OpenAI**, and **FastAPI** with a custom frontend to deliver an interactive, document-aware chatbot.  
It helped me earn my **first $100 as a freelancer.**

---

## 🔍 Demo

🌐 **Live App** → [askakivai.vercel.app](https://askakivai.vercel.app)

📦 **Backend API** (FastAPI) → Hosted on Render  
💬 **Frontend** → Hosted on Vercel (custom HTML + JS)

---

## 📦 Features

- ✅ Custom-trained on client’s fundraising PDFs and DOCX files  
- ✅ Chat history memory with context-aware answers  
- ✅ Vector search using **FAISS** and **OpenAIEmbeddings**  
- ✅ Fallback logic to GPT if vector search is vague  
- ✅ Smart offline/online mode detection with graceful fallback responses  
- ✅ Lightweight frontend with avatars, timestamps, and typing animations  
- ✅ Free-tier deployment (FastAPI + AWS +Vercel)  

---

## 🛠️ Stack

| Component         | Tech Used                        |
|------------------|----------------------------------|
| Vector Store      | `FAISS` (local)                  |
| Embeddings        | `OpenAIEmbeddings`               |
| RAG Chain         | `LangChain` (`ConversationalRetrievalChain`) |
| LLM               | `OpenAI GPT-3.5`                 |
| API Framework     | `FastAPI`                        |
| Frontend          | Custom HTML/CSS/JS               |
| Deployment (API)  | AWS EC2(Free Tier)               |
| Deployment (UI)   | Vercel                           |

---





