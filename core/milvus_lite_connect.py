from pymilvus import MilvusClient
import numpy as np
from core.embbeding_model import get_embedding

class MilvusClientManager:
    def __init__(self, db_path=None, collection_name='test', dimension=512):
        self.collection_name = collection_name
        self.client = MilvusClient(db_path)
        self.dimension = dimension

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )

     # 根据内容生成唯一长int id
    def generate_unique_id(self, content):
        # 使用内容的哈希值作为基础
        hash_value = hash(content)
        
        # 确保ID为正数
        unique_id = abs(hash_value)
        
        # 将ID转换为64位整数
        unique_id = unique_id & ((1 << 64) - 1)
        
        return unique_id

    def insert_data(self, data):
        # 为每条数据添加或替换 id
        for item in data:
            if 'id' not in item or not item['id']:
                # 如果没有 id 或 id 为空，则生成新的 id
                item['id'] = self.generate_unique_id(item.get('text', ''))
            else:
                # 如果已有 id，确保它是长整型
                item['id'] = int(item['id']) & ((1 << 64) - 1)

        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        print(f"Inserted {len(data)} records into collection '{self.collection_name}'.")
        return res

    def search_vectors(self, query, output_field=["text"],  filter_criteria="", limit=6):
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

    def  delete_data(self, filter_criteria="subject == 'history'"):
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


