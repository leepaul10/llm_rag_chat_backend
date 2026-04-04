from fastapi import FastAPI
from rag.retriever import retrieve
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()
api_key=os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set")
client=Groq(api_key=api_key)
SYSTEM_PROMPT="""You are a helpful, accurate, and honest assistant. Follow these rules strictly:

1. **Truthfulness First**: Always prioritize accuracy over providing an answer. Never guess or make up information.

2. **Using Context**: When RAG context is provided, use it as your primary source of truth. Rely on it heavily.

3. **Uncertainty**: If you're not sure about something:
   - Say "I'm not certain about that"
   - Ask clarifying questions: "Could you provide more details about...?"
   - Never fabricate facts

4. **Out of Scope**: If a question is outside your knowledge or the provided context:
   - Say "I don't have information about that"
   - Suggest what information would help: "This would require knowledge about..."

5. **Clarity**: If the question is ambiguous or unclear:
   - Ask the user to clarify: "Do you mean...?" or "Could you elaborate on...?"
   - Provide multiple interpretations if relevant

6. **Concise & Accurate**: Be brief, but complete. Better to be slightly longer and accurate than short and wrong.

7. **Confidence Markers**: Use phrases like:
   - "Based on the provided context..." (when using RAG)
   - "To my knowledge..." (when not using RAG)
   - "I'm confident about..." vs "I'm less certain about..."

8. **Never Hallucinate**: It's better to ask questions or admit uncertainty than to make something up.

Be helpful, but always be honest first."""

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)
conversation_history = []
class ChatRequest(BaseModel):
    message: str
def get_bot_response(user_message):
    try:
        context, score, use_rag=retrieve(user_message)
        print("RAG SCORE:", score)
        if use_rag:
            user_prompt=f"""You have the following context available:
Context:
{context}

Question: {user_message}

Instructions:
1. If the context answers the question, use it with "Based on provided context..."
2. If the context is NOT relevant to the question, IGNORE it and use your general knowledge
3. Always provide the best answer, whether from context or general knowledge"""
            print("USING RAG + Smart Fallback")
        else:
            user_prompt=user_message
            print("USING LLM ONLY")
        #Append original message to conversation history
        conversation_history.append({"role": "user", "content": user_message})
        #Build messages for LLM
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversation_history[-6:]
        #Replace the last user message with the one that includes RAG context if needed
        messages[-1]["content"]=user_prompt
        chat_completion=client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        assistant_reply=chat_completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        prefix="🔍RAG:  " if use_rag else "💡LLM:  "
        return prefix + assistant_reply
    except Exception as e:
        print("ERROR:", e)
        return f"Error: {str(e)}"
@app.post("/chat")
def chat(request: ChatRequest):
    return {"reply": get_bot_response(request.message)}
@app.get("/")
def root():
    return {"message": "RAG + LLM is running!"}
