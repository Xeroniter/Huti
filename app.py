import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import tensorflow.keras.utils as utils

from fastapi import FastAPI, UploadFile, File

import components, gdown

app = FastAPI(title="Hantu Model")

labeler = ['Kertas', 'Batu', 'Gunting']

@app.on_event("startup")
async def imload():
  url = "https://drive.google.com/uc?id=1Mdw85HHcNM6d5yb7rgN7PP6CmevTyNzs"
  if not os.path.isfile("interi.h5"):
    components.inload(url)
  global model
  model = tf.keras.models.load_model("interi.h5", compile=False)

@app.get("/")
async def imroot():
  return {"result":"Hi !"}

@app.post("/logits")
async def predictor(iurl : UploadFile = File(...)):
  rimage = iurl.file.read()
  images = components.imaloader(rimage, (160, 160))
  result = model.predict(images)
  result = labeler[tf.math.argmax(result, axis=1).numpy()[0]]
  return {"result":result}