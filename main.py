import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ml import obtain_image


app = FastAPI()

@app.get("/generate")
def generate_image(prompt:str):
    image = obtain_image(prompt)
    memory_file = io.BytesIO()
    image.save(memory_file, "PNG")
    memory_file.seek(0)
    return StreamingResponse(memory_file, media_type="image/png")
