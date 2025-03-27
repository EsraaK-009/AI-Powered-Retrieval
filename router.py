from fastapi import APIRouter
from generation_chain import get_rag_chain
from pydantic import BaseModel

rag_router = APIRouter()
rag = get_rag_chain()

class ChatMessage(BaseModel):
    question: str


@rag_router.post("/openfoodrag/")
async def get_answer(data: ChatMessage):
    answer = rag.invoke({"question":data.question})
    print(answer['context'])
    return {"query": answer["query"], "LLM_answer": answer["final_answer"]}
