from fastapi import APIRouter 


router1 = APIRouter(prefix='/health',tags=['Healthy'])


@router1.get("/")
def checking_server():
    return {"status":"Healthy","code":200}

