from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os

# =========================
# CONFIG
# =========================
HF_API_KEY = os.environ.get("HF_API_KEY")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita"
MAX_MEMORY = 100  # keep last 10 messages

# =========================
# APP SETUP
# =========================
app = FastAPI(title="Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

# =========================
# 🧠 In-memory chat memory
# (resets when server restarts)
# =========================
messages = [
    {
        "role": "system",
        "content": "You are a friendly, conversational AI assistant who remembers context."
    }
]

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(req: ChatRequest):
    global messages

    # Add user message
    messages.append({"role": "user", "content": req.message})

    # Limit memory
    if len(messages) > MAX_MEMORY:
        messages = [messages[0]] + messages[-MAX_MEMORY:]

    def generate():
        full_reply = ""

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            # SAFETY CHECKS (VERY IMPORTANT)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            content = delta.content
            if not content:
                continue

            full_reply += content
            yield content

        # Save assistant message AFTER streaming finishes
        messages.append({"role": "assistant", "content": full_reply})


    return StreamingResponse(generate(), media_type="text/plain")


# =========================
# RESET MEMORY (OPTIONAL)
# =========================
@app.post("/reset")
def reset_chat():
    global messages
    messages = [
        {
            "role": "system",
            "content": "You are a friendly, conversational AI assistant who remembers context."
        }
    ]
    return {"status": "chat memory cleared"}

@app.get("/")
def root():
    return {"status": "Kyla backend is running"}

