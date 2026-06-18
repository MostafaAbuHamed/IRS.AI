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
        f"Analyze all the provided infrastructure images. They are multiple photos of the same damage. "
        f"Return a JSON object with exactly two fields: "
        f"\"classification\" (must be exactly one of these categories: {categories}) "
        f"and \"description\" (3-5 sentences describing the damage based on all images). "
        f"Return only the JSON, no extra text or markdown."
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
            temperature=0.2,
            max_tokens=1024
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NVIDIA NIM AI service error: {str(e)}"
        )

    raw_text = response.choices[0].message.content
    print(f"NVIDIA NIM raw response: {raw_text}")

    if not raw_text or raw_text.strip() == "":
        raise HTTPException(status_code=500, detail="NVIDIA NIM returned an empty response")

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()

    # Attempt to parse as JSON first
    try:
        result = json.loads(cleaned)
        # Ensure it maps properly to 'classification'
        if "category" in result and "classification" not in result:
            result["classification"] = result.pop("category")
    except json.JSONDecodeError:
        # Fallback to Regex Parsing if the LLM outputted raw Markdown instead of JSON
        category_match = re.search(r"(?:\*?\*?Category\*?\*?|\*?\*?Classification\*?\*?):\s*([^\n\r]+)", raw_text, re.IGNORECASE)
        description_match = re.search(r"(?:\*?\*?Description\*?\*?):\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)

        if category_match and description_match:
            result = {
                "classification": category_match.group(1).strip().strip("*").strip().strip('"').strip("'"),
                "description": description_match.group(1).strip().strip("*").strip()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse NVIDIA NIM response: {raw_text}"
            )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)