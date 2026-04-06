from fastapi import FastAPI
from rag.retriever import retrieve
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

SYSTEM_PROMPT = """You are Tensor, an intelligent AI assistant.

Rules:
- Be clear, direct, and natural.
- Do NOT say things like "based on provided context".
- If unsure, ask for clarification briefly.
- Do NOT over-explain your reasoning.
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)

conversation_history = []

def reset_conversation():
    global conversation_history
    conversation_history = []

class ChatRequest(BaseModel):
    message: str


# STEP 1: CLASSIFICATION FUNCTION
def classify_query(user_message):
    try:
        prompt = f"""
Classify the user query into ONE of these categories:
1. AMBIGUOUS - multiple meanings or unclear intent
2. RAG - needs specific or external data
3. LLM - general knowledge

Return ONLY one word: AMBIGUOUS, RAG, or LLM
Query: {user_message}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        decision = response.choices[0].message.content.strip().upper()

        if decision not in ["AMBIGUOUS", "RAG", "LLM"]:
            return "LLM"

        return decision

    except Exception as e:
        print("CLASSIFICATION ERROR:", e)
        return "LLM"


# MAIN RESPONSE FUNCTION
def get_bot_response(user_message):
    global conversation_history

    try:
        # STEP 2: CLASSIFY QUERY
        decision = classify_query(user_message)
        print("INITIAL DECISION:", decision)

        # STEP 3: HANDLE AMBIGUITY
        if decision == "AMBIGUOUS":
            return f"Your query seems ambiguous: '{user_message}'. Could you clarify what you mean?"

        # STEP 4: PREPARE PROMPT
        user_prompt = user_message  # default

        if decision == "RAG":
            context, score, use_rag = retrieve(user_message)
            print("RAG SCORE:", score)

            if not use_rag or score < 0.6:
                decision = "LLM"
            else:
                user_prompt = f"""
Context:
{context}

Question: {user_message}

Answer ONLY using the context above.
If the answer is not in the context, say you don't know.
"""

        print("FINAL DECISION:", decision)

        # STEP 5: STORE CLEAN USER MESSAGE
        conversation_history.append({"role": "user", "content": user_message})

        # STEP 6: BUILD MESSAGES
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:]

        # ✅ CRITICAL FIX: inject actual prompt (with RAG if needed)
        messages.append({"role": "user", "content": user_prompt})

        # STEP 7: GET RESPONSE
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        assistant_reply = chat_completion.choices[0].message.content

        # STEP 8: STORE RESPONSE
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # STEP 9: LABEL OUTPUT
        prefix = "🔍RAG: " if decision == "RAG" else "💡LLM: "

        return prefix + assistant_reply

    except Exception as e:
        print("ERROR:", e)
        return f"Error: {str(e)}"


@app.post("/clear")
def clear_chat():
    reset_conversation()
    return {"message": "Chat cleared"}


@app.post("/chat")
def chat(request: ChatRequest):
    return {"reply": get_bot_response(request.message)}


@app.get("/")
def root():
    return {"message": "RAG + LLM system is running!"}