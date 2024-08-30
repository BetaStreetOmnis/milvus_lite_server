from core.embbeding_model import get_embedding
from core.milvus_lite_connect import MilvusClientManager
from core.rerank_model import reranker
import os
import configparser

config = configparser.ConfigParser()

dbsite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")

config.read(os.path.join(dbsite_path, 'config.ini'))
secret_key = config.get('api', 'key')

def validate_key(items): 
    key = items["key"] 
    if key != secret_key: 
        return False 
    return True

# 向量转换
def vector_list_main(items):
    # 验证信息
    if not validate_key(items):
        return {"code":422, "msg":"验证失败"}
    vector_list = items['vector_list']
    vectors = [get_embedding(x) for x in vector_list]
    return vectors


def vector_main(items):
    # 验证信息
    if not validate_key(items):
        return {"code":422, "msg":"验证失败"}
    vector = get_embedding(items['text'])
    return vector

# 数据写入
def milvus_insert_main(items):
    if not validate_key(items):
        return {"code":422, "msg":"验证失败"}
    collection_name = items.get("collection_name", "")
    insert_data = items.get("insert_data", "") #[{"text":""}]
    manager = MilvusClientManager(db_path=os.path.join(dbsite_path, collection_name + '.db'), collection_name=collection_name)
    # 向量化text
    for line in insert_data:
        line['vector'] = get_embedding(line['text'])
    manager.insert_data(insert_data)

# 数据查询
def milvus_select_main(items):
    if not validate_key(items):
        return {"code":422, "msg":"验证失败"}
    collection_name = items.get("collection_name", "")
    query = items.get("query", "")
    output_field = items.get("output_field", "")
    manager = MilvusClientManager(db_path=os.path.join(dbsite_path, collection_name + '.db'), collection_name=collection_name)
    res = manager.search_vectors(query, output_field=output_field)
    # print(res[0])
    return res[0]


# 数据删除

# def milvus_delete_main(items):
#     if not validate_key(items):
#         return {"code": 422, "msg": "验证失败"}
#     try:
#         # 假设 collection_name 是集合的名字
#         collection_name = items["collection_name"]
#         conditions = items["conditions"]
#         manager = MilvusClientManager(collection_name=collection_name)
        
#         # 构建删除条件（表达式），这里假设 items 是一个包含条件的字典
#         # 例如，items = {"field_name": "value"}
#         for field, value in items.items():
#             conditions.append(f'{field} == "{value}"')
#         expr = " and ".join(conditions)

#         # 执行删除操作
#         delete_result = manager.delete(expr=expr)
        
#         if delete_result.status.code == 0:
#             return {"code": 200, "msg": "删除成功"}
#         else:
#             return {"code": 500, "msg": f"删除失败: {delete_result.status.message}"}
    
#     except Exception as e:
#         return {"code": 500, "msg": f"内部错误: {str(e)}"}
    

# 数据查询
def milvus_search_main(items):
    filter_criteria = items.get("filter_criteria", "")
    query = items.get("query")
    collection_name = items["collection_name"]
    manager = MilvusClientManager(db_path=os.path.join(dbsite_path, collection_name + '.db'), collection_name=collection_name)
    output_field = items.get("output_field", "")
    limit = items.get("limit", 10)
    rerank = items.get("rerank", 0)
    res = manager.search_vectors(query, output_field=output_field,  filter_criteria=filter_criteria, limit=limit)
    print(res)
    # ["[{'id': 23, 'distance': 0.8208538889884949, 'entity': {'text': '测试数据', 'id': 23}}, {'id': 22, 'distance': 0.8208538889884949, 'entity': {'text': '测试数据', 'id': 22}}]"] , extra_info: {'cost': 0}
    res = res[0]
    # 是否开启rerank 重排序
    if rerank:
        document = [x['entity']['text'] for x in res]
        rerank_res = reranker(query, document)
        print(rerank_res)
        for idx, similar_percent in enumerate(rerank_res):
            res[idx]["rerank_score_percent"] = similar_percent[-1]
    return res


# if __name__ == "__main__":
#     # milvus_search_main({""})
#     milvus_insert_main({"key":"367686", "insert_data":[{"id":23, "text":"haohao"}], "collection_name":"test"})
    # res = milvus_search_main({"key":"367686", "query": "测试", "collection_name":"test", "limit":2, "output_field":["id", "text"], "rerank":1})
    # print(res)