from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------- 请求体模型 ----------
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

# ---------- 路由 ----------
# 1. 根路径，测试服务是否启动
@app.get("/")
def read_root():
    return {"message": "Welcome to Agent API"}

# 2. 路径参数：获取单个用户
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User_{user_id}"}

# 3. 路径参数 + 查询参数：获取用户帖子
@app.get("/users/{user_id}/posts")
def get_user_posts(
    user_id: int,
    published: bool = True,
    limit: int = 10
):
    return {
        "user_id": user_id,
        "published": published,
        "limit": limit,
        "posts": [f"post_{i}" for i in range(limit)]
    }

# 4. POST 请求体：创建物品
@app.post("/items/")
def create_item(item: Item):
    # 实际项目中会保存到数据库
    return {
        "message": "Item created",
        "item_name": item.name,
        "price_with_tax": item.price * 1.1
    }

# 5. 模拟 Agent 任务执行（组合路径参数 + 请求体）
class TaskRequest(BaseModel):
    task: str
    context: list[str] = []

@app.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, request: TaskRequest):
    # 这里将来可以接大模型 API
    return {
        "agent_id": agent_id,
        "task": request.task,
        "context": request.context,
        "result": f"Agent {agent_id} completed task: {request.task}"
    }
