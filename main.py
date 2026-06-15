import os
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    categories = [
        "Damaged concrete structures",
        "DamagedElectricalPoles",
        "DamagedRoadSigns",
        "DeadAnimalsPollution",
        "FallenTrees",
        "Garbage",
        "Graffitti",
        "IllegalParking",
        "Potholes and RoadCracks"
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
            f"Analyze this infrastructure image. Return a JSON object with exactly two fields: "
            f"\"classification\" (must be exactly one of these categories: {categories}) "
            f"and \"description\" (2-3 sentences describing the damage in detail). "
            f"Return only the JSON, no extra text or markdown."
        ]
    )

    raw_text = response.text

    # Log it so we can see what Gemini actually returned
    print(f"Gemini raw response: {raw_text}")

    if not raw_text or raw_text.strip() == "":
        raise HTTPException(status_code=500, detail="Gemini returned an empty response")

    # Strip markdown code blocks if Gemini wrapped the JSON in them
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()

    result = json.loads(cleaned)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)