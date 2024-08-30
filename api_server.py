from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
# from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
# from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from base_server import common_logic
from fastapi import FastAPI, Body, HTTPException, Header
from pydantic import BaseModel
from fastapi import FastAPI

# from core.rerank_model import reranker
# from core.embbeding_model import get_embedding
import os
import sys
# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execute_main import vector_list_main, vector_main, milvus_insert_main, milvus_select_main, milvus_search_main,reranker,get_embedding
from base_server import common_logic


app = FastAPI(title='ai检索能力组件',description='ai基础能力接口')

templates_path = os.path.join(os.path.dirname(__file__), "templates")

templates = Jinja2Templates(templates_path)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，也可以指定特定的来源
    allow_credentials=True,  # 允许携带身份验证信息（例如，Cookies）
    allow_methods=["*"],  # 允许所有HTTP方法（GET、POST、PUT、DELETE等）
    allow_headers=["*"],  # 允许所有HTTP头部
)

# 单个text 向量化
@app.post("/vector/",summary='数据向量化',description='',tags=['算法'])
async def simhash(item:dict): return common_logic(item, vector_main)


# 数据向量化存储
@app.post("/vectors/",summary='数据向量化列表',description='',tags=['算法'])
async def simhash(item:dict): return common_logic(item, vector_list_main)

# 向量数据入库（主键不可重复）
@app.post("/milvus_insert/",summary='数据向量化存储',description='',tags=['算法'])
async def simhash(item:dict): return common_logic(item, milvus_insert_main)

# 向量数据查询
@app.post("/milvus_select/",summary='数据向量化查询',description='',tags=['算法'])
async def simhash(item:dict): return common_logic(item, milvus_search_main)

# # 向量数据删除
# @app.post("/milvus_delete/",summary='数据向量化查询',description='',tags=['算法'])
# async def simhash(item:dict): return common_logic(item, milvus_delete_main)

# 向量数据重排序rerank
class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]

class RerankDocument(BaseModel):
    document: str
    score: float

class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankDocument]

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        # 调用你的 rerank 逻辑，并返回排序结果
        ranked_documents = reranker(request.query, request.documents)
        return RerankResponse(data=ranked_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# openai type  embedding api
class EmbeddingRequest(BaseModel):
    input: List[str]

class EmbeddingResponse(BaseModel):
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    # 获取输入文本
    input_texts = request.input
    # 构造与OpenAI类似的API返回格式
    result = EmbeddingResponse(
        data=[{"embedding": get_embedding(emb), "index": idx} for idx, emb in enumerate(input_texts)],
        model="GTE-to-OpenAI-adapted",
        usage={"total_tokens": len(input_texts)}  # 模拟token消耗
    )
    return result


# dify对接  知识库检索工具对接api

class InputData(BaseModel):
    point: str
    params: dict = {}

@app.post("/dify_milvus_search")
async def dify_receive(data: InputData = Body(...), authorization: str = Header(None)):
    """
    Receive API query data from Dify.
    """
    expected_api_key = "367686"  # TODO Your API key of this API
    auth_scheme, _, api_key = authorization.partition(' ')
    if auth_scheme.lower() != "bearer" or api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    point = data.point
    if point == "ping":
        return {
            "result": "pong"
        }
    if point == "app.external_data_tool.query":
        res =  milvus_search_main(params=data.params["inputs"])
        return res
    raise HTTPException(status_code=400, detail="Not implemented")


# def handle_app_external_data_tool_query(params: dict):
#     try:
#         keyword = params['inputs']['query']
#     except Exception as e:
#         keyword = params['query']
#     payload = {}
#     headers = {}
#     x = bing_search(keyword)

#     x = x.replace('"','').replace("'","")
#     write_sql(f"insert into google_search (content) values ('{x}')")
#     return {"result":x}





if __name__ == '__main__':
    uvicorn.run("api_server:app", host="0.0.0.0", port=8088, reload=False)