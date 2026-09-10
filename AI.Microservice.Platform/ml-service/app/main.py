# class my_class(object):
#     pass


from fastapi import FastAPI
from app.model.predict import predict
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict([data.feature1, data.feature2, data.feature3])
    return {"prediction": int(result)}

# Add this at the absolute end of your main.py file
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

