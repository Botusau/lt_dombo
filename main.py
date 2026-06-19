import uvicorn
from fastapi import FastAPI

from app.endpoints import router as endpoints_router

app = FastAPI(
    title="LTAutoML API",
    description="API для автоматического машинного обучения с использованием LightAutoML. "
                "Поддерживает задачи классификации и регрессии."
)

app.include_router(endpoints_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
