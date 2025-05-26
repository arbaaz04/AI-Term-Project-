from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pathlib
import re
import gdown
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

model_path = str(ROOT_DIR / "trained_models" / "flan-t5-base-medical-chatbot-finetuned")
file_id = "1NzLDdHLAWUjgQXSz___h0O5eFgzL7nDf"
safetensors_file_path = model_path  + "\\model.safetensors"
if not os.path.exists(safetensors_file_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from: {url}")
    gdown.download(url, output=safetensors_file_path, quiet=False)
else:
    print(f"Model file already exists at: {safetensors_file_path}")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def filter_sentences(text):
    if not text or not text.strip():
        return text

    sentences = re.split(r'[.!?]+', text)
    filtered_sentences = []

    filter_patterns = [
        r'regards',
        r'wish you a very',
        r'i have read your question',
        r'thanks for'
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            should_filter = False
            for pattern in filter_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    should_filter = True
                    break

            if not should_filter:
                filtered_sentences.append(sentence)

    if filtered_sentences:
        result = '. '.join(filtered_sentences)
        if text.rstrip() and text.rstrip()[-1] in '.!?':
            result += '.'
        return result
    else:
        return "I apologize, but I cannot provide a specific response to that query."

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(str(ROOT_DIR / "static" / "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

class ChatInput(BaseModel):
    message: str

    @validator("message")
    def validate_message(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value

@app.post("/chat")
async def chat(user_input: ChatInput):
    inputs = tokenizer(user_input.message, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    filtered_response = filter_sentences(response)
    return {"response": filtered_response}
