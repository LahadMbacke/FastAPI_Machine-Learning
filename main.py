import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ml_images import obtain_image
from ml_ner import extract_entities


app = FastAPI()

@app.get("/generate")
def generate_image(prompt:str):
    image = obtain_image(prompt)
    memory_file = io.BytesIO()
    image.save(memory_file, "PNG")
    memory_file.seek(0)
    return StreamingResponse(memory_file, media_type="image/png")


@app.get("/ner")
def ner(text:str):
    entites = extract_entities(text)
    return entites


