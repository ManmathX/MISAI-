import os
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create router for chat endpoints
router = APIRouter()


# ================================
# � MODELS
# ================================

class ConversationItem(BaseModel):
    text: str
    sender: str


class QueryInput(BaseModel):
    message: str
    conversation_history: List[ConversationItem]


# ================================
# 🤖 MISBOT ENDPOINT
# ================================

@router.post("/misbot")
def misbot(input: QueryInput):
    """
    Chatbot endpoint using Gemini to check statement accuracy.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}

        # Prepare conversation context for Gemini
        history = []
        for msg in input.conversation_history:
            role = "user" if msg.sender == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg.text}]})

        # Add the latest message
        history.append({
            "role": "user",
            "parts": [{"text": f"Check the accuracy and authenticity of this statement:\n\n{input.message}"}]
        })

        payload = {"contents": history}

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        # Extract the Gemini model's text response safely
        ai_reply = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response from Gemini.")
        )

        return {"result": ai_reply}

    except Exception as e:
        return {"error": str(e)}