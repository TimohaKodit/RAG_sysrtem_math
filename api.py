from fastapi import FastAPI
from pydantic import BaseModel
from core import get_rag_chain



class Query(BaseModel):
    input: str
    user_id: int
    chat_history: list = []

app = FastAPI()



@app.post('/ask')
def invoke(inv: Query):
    chain = get_rag_chain(inv.user_id)
    response = chain.invoke({
        "input": inv.input, 
        
        "chat_history": inv.chat_history
    })

    return {"answer": response['answer']}