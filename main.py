import os
import json
import re
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from openai import OpenAI
from typing import List

load_dotenv()

app = FastAPI()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
    api_key=NVIDIA_API_KEY
)

ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif"]

@app.post("/classify-image")
async def classify_image(images: List[UploadFile] = File(...)):

    # Validate all files first
    for file in images:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WebP, GIF, HEIC, HEIF"
            )

    categories = [
        "Damaged concrete structures",
        "Damaged Electrical Poles",
        "Damaged Road Signs",
        "Dead Animals Pollution",
        "Fallen Trees",
        "Garbage",
        "Graffitti",
        "Illegal Parking",
        "Potholes and Road Cracks",
        "Water leak"
    ]

    prompt = (
        f"Analyze the provided infrastructure images. "
        f"You must respond with ONLY a raw JSON object. "
        f"No markdown, no code blocks, no explanation, no extra text. "
        f"The JSON must have exactly two fields: "
        f"\"category\" (must be exactly one of: {categories}) "
        f"and \"description\" (3-5 sentences describing the damage). "
        f"Example of the exact format you must return: "
        f'{{ "category": "Potholes and Road Cracks", "description": "The road surface has significant damage." }}'
    )

    message_content_list = []

    # Process each image and convert to base64 for OpenAI format
    for file in images:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        message_content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{file.content_type};base64,{base64_image}"
            }
        })

    # Add prompt text to contents
    message_content_list.append({
        "type": "text",
        "text": prompt
    })

    # Call NVIDIA NIM
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": message_content_list
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again later."
        )

    raw_text = response.choices[0].message.content
    print(f"NVIDIA NIM raw response: {raw_text}")

    if not raw_text or raw_text.strip() == "":
        raise HTTPException(status_code=500, detail="NVIDIA NIM returned an empty response")

    # Strip any accidental markdown fences just in case
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON: {raw_text}"
        )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)