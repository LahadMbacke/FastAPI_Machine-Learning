import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ml_images import obtain_image
from ml_ner import extract_entities
from pydantic import BaseModel
from typing import List


app = FastAPI()

class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int

@app.get("/generate")
def generate_image(prompt:str):
    image = obtain_image(prompt)
    memory_file = io.BytesIO()
    image.save(memory_file, "PNG")
    memory_file.seek(0)
    return StreamingResponse(memory_file, media_type="image/png")


@app.get("/ner", response_model=List[Entity])
def ner(text_input:str):
    entites = extract_entities(text_input)
    entite_lists = []
    for entity in entites:
        entite_lists.append(Entity(**entity))
    return entite_lists


