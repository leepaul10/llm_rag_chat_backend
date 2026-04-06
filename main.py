# from fastapi import FastAPI
# from rag.retriever import retrieve
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from groq import Groq
# import os
# load_dotenv()
# api_key=os.environ.get("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("GROQ_API_KEY is not set")
# client=Groq(api_key=api_key)
# SYSTEM_PROMPT = """You are Tensor, an intelligent AI assistant powered by Retrieval-Augmented Generation (RAG).

# Your name is Tensor. When asked your name, respond:
# "Hey, I'm Tensor! An AI assistant powered by RAG technology."

# Follow these principles:

# 1. Truthfulness First:
# Always prioritize accuracy. Never fabricate information.

# 2. Use Context When Available:
# If RAG context is provided, use it as a primary source.
# However, do NOT ignore your general knowledge.

# 3. Smart Fallback:
# If the answer is not in the provided context but is a well-known fact,
# use your general knowledge to answer clearly and confidently.

# 4. Handle Uncertainty Carefully:
# Only say "I'm not certain" when the information is truly unknown or ambiguous.
# Do NOT use uncertainty for basic or common knowledge questions.

# 5. Avoid Over-Questioning:
# Only ask clarifying questions when the query is genuinely unclear.
# Do NOT ask for clarification for simple, well-known topics.

# 6. Natural Responses:
# Avoid robotic phrases like:
# "Based on the provided context..."
# Instead, respond naturally and directly.

# 7. Concise & Helpful:
# Keep answers clear, direct, and informative without unnecessary repetition.

# 8. Never Hallucinate:
# If the answer cannot be determined from context or general knowledge,
# clearly say so instead of guessing.

# Your goal is to be helpful, natural, and accurate — not rigid.
# """
# app=FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
#     allow_credentials=False
# )
# conversation_history = []
# conversation_history = []

# def reset_conversation():
#     global conversation_history
#     conversation_history = []
# class ChatRequest(BaseModel):
#     message: str
# def get_bot_response(user_message):
#     try:
#         context, score, use_rag=retrieve(user_message)
#         print("RAG SCORE:", score)
#         # ✅ Check if query is AMBIGUOUS
#         if use_rag and 0.65 < score < 0.69:
#             clarify_prompt = f"User asked: '{user_message}'. This is ambiguous. Ask them to clarify in 1-2 sentences only."
#             messages = [{"role": "system", "content": SYSTEM_PROMPT},
#                        {"role": "user", "content": clarify_prompt}]
#             clarification = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
#             return "" + clarification.choices[0].message.content
 

#         if use_rag:
#             user_prompt=f"""You have the following context available:
# Context:
# {context}

# Question: {user_message}

# Instructions:
# 1. If the context answers the question, use it with "Based on provided context..."
# 2. If the context is NOT relevant to the question, IGNORE it and use your general knowledge
# 3. Always provide the best answer, whether from context or general knowledge
# 4. If ambiguity or confusion occurs ask the user about clarification and proceed with it dont rethink for simple things and try to obey the user"""
#             print("USING RAG + Smart Fallback")
#         else:
#             user_prompt=user_message
#             print("USING LLM ONLY")
#         #Append original message to conversation history
#         conversation_history.append({"role": "user", "content": user_message})
#         #Build messages for LLM
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT}
#         ] + conversation_history[-10:]
#         #Replace the last user message with the one that includes RAG context if needed
#         messages[-1]["content"]=user_prompt
#         chat_completion=client.chat.completions.create(
#             messages=messages,
#             model="llama-3.3-70b-versatile",
#         )
#         assistant_reply=chat_completion.choices[0].message.content
#         conversation_history.append({"role": "assistant", "content": assistant_reply})
#         prefix="🔍RAG:  " if use_rag else "💡LLM:  "
#         return prefix + assistant_reply
#     except Exception as e:
#         print("ERROR:", e)
#         return f"Error: {str(e)}"
# @app.post("/clear")
# def clear_chat():
#     reset_conversation()
#     return {"message": "Chat cleared"}
# @app.post("/chat")
# def chat(request: ChatRequest):
#     return {"reply": get_bot_response(request.message)}
# @app.get("/")
# def root():
#     return {"message": "RAG + LLM is running!"}




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

SYSTEM_PROMPT = """You are Tensor, an intelligent AI assistant powered by Retrieval-Augmented Generation (RAG).

Your name is Tensor. When asked your name, respond:
"Hey, I'm Tensor! An AI assistant powered by RAG technology."

Follow these principles:

1. Truthfulness First:
Always prioritize accuracy. Never fabricate information.

2. Use Context When Available:
If RAG context is provided, use it as a primary source.
However, do NOT ignore your general knowledge.

3. Smart Fallback:
If the answer is not in the provided context but is a well-known fact,
use your general knowledge to answer clearly and confidently.

4. Handle Uncertainty Carefully:
Only say "I'm not certain" when the information is truly unknown or ambiguous.

5. Avoid Over-Questioning:
Only ask clarifying questions when the query is genuinely unclear.

6. Natural Responses:
Do NOT mention "provided context" explicitly. Respond naturally.

7. Concise & Helpful:
Keep answers clear, direct, and informative.

8. Never Hallucinate:
If unsure, say so instead of guessing.

Your goal is to be helpful, natural, and accurate.
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

# 🔥 NEW: LLM ROUTER (decides RAG vs LLM vs Clarify)
def decide_mode(query):
    try:
        prompt = f"""
Classify the query into ONE of these:
- SPECIFIC → needs external or detailed knowledge (use RAG)
- GENERAL → common knowledge (use LLM)
- AMBIGUOUS → unclear meaning (ask clarification)

Query: "{query}"

Answer ONLY one word: SPECIFIC, GENERAL, or AMBIGUOUS
"""
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        return res.choices[0].message.content.strip()

    except Exception as e:
        print("Router Error:", e)
        return "GENERAL"  # safe fallback

def get_bot_response(user_message):
    try:
        mode = decide_mode(user_message)
        print("ROUTER MODE:", mode)

        context = ""
        score = 0
        use_rag = False

        # 🔥 Handle ambiguity FIRST
        if mode == "AMBIGUOUS":
            return "Your question could mean multiple things. Can you clarify?"

        # 🔥 Use RAG only if needed
        if mode == "SPECIFIC":
            context, score, use_rag = retrieve(user_message)
            print("RAG SCORE:", score)

            if score < 0.55:
                use_rag = False

        # 🔥 Build prompt
        if use_rag:
            user_prompt = f"""Context:
{context}

Question: {user_message}

Instructions:
- Use the context only if relevant
- If not useful, ignore it and answer normally
- Answer naturally and clearly
"""
            print("USING RAG")
        else:
            user_prompt = user_message
            print("USING LLM ONLY")

        # Store user message
        conversation_history.append({"role": "user", "content": user_message})

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversation_history[-10:]

        messages[-1]["content"] = user_prompt

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        assistant_reply = chat_completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": assistant_reply})

        prefix = "🔍RAG:  " if use_rag else "💡LLM:  "
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
    return {"message": "RAG + LLM is running!"}