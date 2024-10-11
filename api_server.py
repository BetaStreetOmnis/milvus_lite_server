from fastapi import FastAPI, Body, HTTPException, Header
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import configparser
import traceback

# 添加当前文件路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from core.embbeding_model import get_embedding
from core.milvus_lite_connect import MilvusClientManager
from core.rerank_model import reranker

# 读取配置文件
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))
secret_key = config.get('api', 'key')


dbsite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db')

# 创建FastAPI应用
app = FastAPI(title='语义检索', description='基于gte的语义检索组件')

# 设置模板路径
templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(templates_path)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 验证API密钥
def validate_key(authorization: str):
    # auth_scheme, _, api_key = authorization.partition(' ')
    # return auth_scheme.lower() == "bearer" and api_key == secret_key
    if authorization == secret_key:
        return True
    else:
        return False

# 定义请求模型
class VectorRequest(BaseModel):
    text: str

class VectorListRequest(BaseModel):
    vector_list: List[str]

class MilvusInsertRequest(BaseModel):
    collection_name: str
    insert_data: List[Dict[str, Any]]

class MilvusSearchRequest(BaseModel):
    collection_name: str
    query: str
    output_field: List[str]
    filter_criteria: str = ""
    limit: int = 10
    rerank: int = 0

class MilvusDeleteRequest(BaseModel):
    collection_name: str
    delete_data: List[Dict[str, Any]]

# 数据向量化接口
@app.post("/vector/", summary='数据向量化', description='', tags=['算法'])
async def simhash(request: VectorRequest, authorization: str = Header(None)): 
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        vector = get_embedding(request.text)
        return {"status_code": 200, "data": vector}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc()))

# 数据向量化列表接口
@app.post("/vectors/", summary='数据向量化列表', description='', tags=['算法'])
async def simhash(request: VectorListRequest, authorization: str = Header(None)): 
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        vectors = [get_embedding(x) for x in request.vector_list]
        return {"status_code": 200, "data": vectors}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc()))

# 数据向量化存储接口
@app.post("/milvus_insert/", summary='数据向量化存储', description='', tags=['算法'])
async def simhash(request: MilvusInsertRequest, authorization: str = Header(None)): 
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        manager = MilvusClientManager(db_path=os.path.join(dbsite_path, request.collection_name + '.db'), collection_name=request.collection_name)
        for line in request.insert_data:
            line['vector'] = get_embedding(line['text'])
        manager.insert_data(request.insert_data)
        return {"status_code": 200, "data": "数据插入成功"}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc()))

# 数据向量化查询接口
@app.post("/milvus_select/", summary='数据向量化查询', description='', tags=['算法'])
async def simhash(request: MilvusSearchRequest, authorization: str = Header(None)): 
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        manager = MilvusClientManager(db_path=os.path.join(dbsite_path, request.collection_name + '.db'), collection_name=request.collection_name)
        res = manager.search_vectors(request.query, output_field=request.output_field, filter_criteria=request.filter_criteria, limit=request.limit)
        res = res[0]
        if request.rerank:
            document = [x['entity']['text'] for x in res]
            rerank_res = reranker(request.query, document)
            for idx, similar_percent in enumerate(rerank_res):
                res[idx]["rerank_score_percent"] = similar_percent[-1]
        return {"status_code": 200, "data": res}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc()))

# 数据向量化删除接口
@app.post("/milvus_delete/", summary='数据向量化删除', description='', tags=['算法'])
async def simhash(request: MilvusDeleteRequest, authorization: str = Header(None)): 
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        filter_criteria = " and ".join(
            "{} == '{}'".format(key, value) if isinstance(value, str) else "{} == {}".format(key, value)
            for key, value in request.delete_data[0].items()
        )
        manager = MilvusClientManager(db_path=os.path.join(dbsite_path, request.collection_name + '.db'), collection_name=request.collection_name)
        res = manager.delete_data(filter_criteria=filter_criteria)
        return {"status_code": 200, "data": res}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc()))

# 重排序请求模型
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

# 重排序接口
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest, authorization: str = Header(None)):
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        ranked_documents = reranker(request.query, request.documents)
        return RerankResponse(data=ranked_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 嵌入请求模型
class EmbeddingRequest(BaseModel):
    input: List[str]

class EmbeddingResponse(BaseModel):
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

# 创建嵌入接口
@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, authorization: str = Header(None)):
    try:
        if not validate_key(authorization):
            return {"code": 422, "msg": "验证失败"}
        input_texts = request.input
        result = EmbeddingResponse(
            data=[{"embedding": get_embedding(emb), "index": idx} for idx, emb in enumerate(input_texts)],
            model="GTE-to-OpenAI-adapted",
            usage={"total_tokens": len(input_texts)}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 输入数据模型
class InputData(BaseModel):
    point: str
    params: dict = {}

# Dify Milvus 搜索接口
@app.post("/dify_milvus_search")
async def dify_receive(data: InputData = Body(...), authorization: str = Header(None)):
    expected_api_key = "367686"
    auth_scheme, _, api_key = authorization.partition(' ')
    if auth_scheme.lower() != "bearer" or api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    point = data.point
    if point == "ping":
        return {"result": "pong"}
    if point == "app.external_data_tool.query":
        items = data.params["inputs"]
        filter_criteria = items.get("filter_criteria", "")
        query = items.get("query")
        collection_name = items["collection_name"]
        manager = MilvusClientManager(db_path=os.path.join(dbsite_path, collection_name + '.db'), collection_name=collection_name)
        output_field = items.get("output_field", "")
        limit = items.get("limit", 10)
        rerank = items.get("rerank", 0)
        res = manager.search_vectors(query, output_field=output_field, filter_criteria=filter_criteria, limit=limit)
        print(res)
        res = res[0]
        if rerank:
            document = [x['entity']['text'] for x in res]
            rerank_res = reranker(query, document)
            print(rerank_res)
            for idx, similar_percent in enumerate(rerank_res):
                res[idx]["rerank_score_percent"] = similar_percent[-1]
        return res
    raise HTTPException(status_code=400, detail="Not implemented")

# 启动应用
if __name__ == '__main__':
    uvicorn.run("api_server:app", host="0.0.0.0", port=8089, reload=False)