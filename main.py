import os
import json
import re
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types, errors as genai_errors
from typing import List

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

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

    # Build contents list — one Part per image + the prompt at the end
    contents = []
    for file in images:
        image_bytes = await file.read()
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=file.content_type))

    contents.append(
        f"Analyze all the provided infrastructure images. They are multiple photos of the same damage. "
        f"Return a JSON object with exactly two fields: "
        f"\"classification\" (must be exactly one of these categories: {categories}) "
        f"and \"description\" (3-5 sentences describing the damage based on all images). "
        f"Return only the JSON, no extra text or markdown."
    )

    # Call Gemini with error handling
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents
        )
    except genai_errors.ServerError:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    raw_text = response.text
    print(f"Gemini raw response: {raw_text}")

    if not raw_text or raw_text.strip() == "":
        raise HTTPException(status_code=500, detail="Gemini returned an empty response")

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()

    # Parse JSON with error handling
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Gemini response as JSON: {cleaned}"
        )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)