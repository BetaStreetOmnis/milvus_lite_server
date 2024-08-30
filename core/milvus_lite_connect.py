from pymilvus import MilvusClient
import numpy as np
from core.embbeding_model import get_embedding

class MilvusClientManager:
    def __init__(self, db_path="/root/milvus_lite_server/milvus_demo5.db", collection_name='test', dimension=512):
        self.collection_name = collection_name
        self.client = MilvusClient(db_path)
        self.dimension = dimension

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )

    def insert_data(self, data):
        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        print(f"Inserted {len(data)} records into collection '{self.collection_name}'.")
        return res

    def search_vectors(self, query, output_field=["text"],  filter_criteria="", limit=10):
        # 搜索与 query_vector 最接近的向量
        normalized_query_embedding = get_embedding(query)
        res = self.client.search(
            collection_name=self.collection_name,
            data=[normalized_query_embedding],
            filter=filter_criteria,
            limit=limit,
            output_fields=output_field
        )
        return res

    def query_data(self, filter_criteria="subject == 'history'"):
        # 根据过滤条件查询数据
        res = self.client.query(
            collection_name=self.collection_name,
            filter=filter_criteria,
            output_fields=["text", "subject"]
        )
        print(f"Query result: {res}")
        return res

    def delete_data(self, filter_criteria="subject == 'history'"):
        # 根据过滤条件删除数据
        res = self.client.delete(
            collection_name=self.collection_name,
            filter=filter_criteria
        )
        print(f"Deleted records with filter '{filter_criteria}'.")
        return res
    



def milvus_insert_main(items):
    collection_name = items.get("collection_name", "")
    insert_data = items.get("insert_data", "") #[{"text":""}]
    manager = MilvusClientManager(collection_name=collection_name)
    # 向量化text
    for line in insert_data:
        line['vector'] = get_embedding(line['text'])
    # print(insert_data)
    # for entity in insert_data:
    #     print("Entity vector shape:", entity["vector"].shape)
    #     print("Entity content:", entity)
    manager.insert_data(insert_data)


def milvus_select_main(items):
    collection_name = items.get("collection_name", "")
    query = items.get("query", "")
    output_field = items.get("output_field", "")
    manager = MilvusClientManager(collection_name=collection_name)
    res = manager.search_vectors(query, output_field=output_field)
    return res




# # # 使用示例
# if __name__ == "__main__":
#     print(1)
#     res = milvus_select_main({"collection_name":"demo_collection1", "query":"你好", "output_field":["text", "title"]})
#     print(res)
#     # milvus_insert_main({"collection_name":"demo_collection1", "insert_data":[{"text":"666", "id":1, "title":"ddd"}, {"text":"6662223345ddsfsdgfs", "id":2, "title":"33333d"}]})
# #     # 初始化管理器
# #     manager = MilvusClientManager(db_path="/root/milvus_lite_server/milvus_demo5.db", collection_name="demo_collection")

# #     # 创建集合

# #     # 示例数据
# #     docs = [
# #         "Artificial intelligence was founded as an academic discipline in 1956.",
# #         "Alan Turing was the first person to conduct substantial research in AI.",
# #         "Born in Maida Vale, London, Turing was raised in southern England."
# #     ]
# #     vectors = [np.array(get_embedding(x), dtype=np.float32) for x in docs]
# #     vectors = [ (x / np.linalg.norm(x))[:512]  for x in vectors ]
# #     data = [ {"id": i+5, "vector": vectors[i].tolist(), "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
# #     # 搜索数据
# #     manager.search_vectors('intell')


#     #


# # from pymilvus import MilvusClient
# # import numpy as np
# # from embbeding_model import get_embedding

# # client = MilvusClient("./milvus_demo6.db")
# # client.create_collection(
# #     collection_name="demo_collection",
# #     dimension=512  # The vectors we will use in this demo has 384 dimensions
# # )

# # docs = [
# #     "Artificial intelligence was founded as an academic discipline in 1956.",
# #     "Alan Turing was the first person to conduct substantial research in AI.",
# #     "Born in Maida Vale, London, Turing was raised in southern England.",
# # ]

# # # vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
# # vectors = [np.array(get_embedding(x), dtype=np.float32) for x in docs]
# # vectors = [ (x / np.linalg.norm(x))[:512]  for x in vectors ]
# # data = [ {"id": i+5, "vector": vectors[i].tolist(), "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
# # res = client.insert(
# #     collection_name="demo_collection",
# #     data=data
# # )

# # res = client.search(
# #     collection_name="demo_collection",
# #     data=[vectors[0]],
# #     filter="subject == 'history'",
# #     limit=2,
# #     output_fields=["text", "subject"],
# # )
# # print(res)

# # res = client.query(
# #     collection_name="demo_collection",
# #     filter="subject == 'history'",
# #     output_fields=["text", "subject"],
# # )
# # print(res)

# # res = client.delete(
# #     collection_name="demo_collection",
# #     filter="subject == 'history'",
# # )
# # print(res)
