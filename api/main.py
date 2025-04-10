import uvicorn
from fastapi import FastAPI
from api.routing.predict import router

app = FastAPI()


# endpoint для отладки
@app.get("/")
def read_root():
    return {"Hello": "from FastAPI"}


app.include_router(router)

# Запускаем сервер при непосредственном выполнении скрипта
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
