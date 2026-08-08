from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(title=settings.app_name)


@app.get("/")
async def root():
    return {"app": settings.app_name, "Status" : "ok"}

