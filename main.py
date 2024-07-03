from fastapi import FastAPI
from ml import obtain_image


app = FastAPI()

@app.get("/generate")
def generate_image(prompt:str):
    image = obtain_image(prompt)
    return image
