from fastapi import FastAPI
from backend.rag.retriever import retrieve
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

client = Groq(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False  # Fixed: cannot use wildcard origins with credentials=True
)

conversation_history = []

class ChatRequest(BaseModel):
    message: str

def get_bot_response(user_message):
    try:
        context, score, use_rag = retrieve(user_message)
        print("RAG SCORE:", score)

        if use_rag:
            user_prompt = f"""Use the context below to answer the question.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question:
{user_message}"""
            print("USING RAG")
        else:
            user_prompt = user_message
            print("USING CHAT")

        conversation_history.append({"role": "user", "content": user_prompt})

        messages = [
            {"role": "system", "content": "You are a helpful assistant, concise and informative."}
        ] + conversation_history[-6:]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        assistant_reply = chat_completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        prefix = "𝄜Sources:  " if use_rag else "֎LLM:  "
        return prefix + assistant_reply

    except Exception as e:
        print("ERROR:", e)
        return f"Error: {str(e)}"

@app.post("/chat")
def chat(request: ChatRequest):
    return {"reply": get_bot_response(request.message)}
