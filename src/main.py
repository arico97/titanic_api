import os
from train import MakeTitanicPrediction
from data_analysis import Datanalysizer
from fastapi import APIRouter, Request, FastAPI 


app = FastAPI()
api_router = APIRouter()

@api_router.get("/", status_code=200)
def version(): 
    return {"version": "V0"}

@api_router.post("/analysis")
async def predict(request: Request):
    body = await request.json()
    train_path = body["train_path"]
    test_path = body["test_path"]
    return {"success" : True}


@api_router.post("/predict")
async def predic(request: Request): 
    body = await request.json()
    train_path = body["train_path"]
    test_path = body["test_path"]
    

    print('start training endpoint')
    outupt_path=MakeTitanicPrediction(train_path,test_path).path_out_write()
    print(outupt_path)
    return {"success": True,
            "outupt path":outupt_path}

app.include_router(api_router)
