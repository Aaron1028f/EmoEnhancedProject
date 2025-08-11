from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/random")
async def random_number():
    return {"random_number": random.randint(1, 100)}

@app.get("/random/{max_value}")
async def random_number(max_value: int):
    return {"random_number": random.randint(1, max_value)}

# fastapi 教學，@app 的所有用法
# 有.get, @app.post, @app.put, @app.delete

@app.get("/random/yield")
async def random_number_yield():
    def get_num():
        for i in range(5):
            yield i
