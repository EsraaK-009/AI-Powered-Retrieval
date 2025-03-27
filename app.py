from fastapi import FastAPI
from router import rag_router

app = FastAPI()

app.include_router(
    rag_router
)

