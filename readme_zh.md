## 简介
这是一个使用 Milvus Lite 进行向量检索和排序的项目。项目实现了将文本转换为向量、向量数据的插入、检索、以及使用 reranker 进行重排序的功能。项目采用 Python 实现，并支持通过 RESTful API 进行操作。

## 功能
向量转换：将输入文本转换为向量表示。
数据写入：将文本数据向量化后插入 Milvus Lite 数据库。
数据查询：从 Milvus Lite 数据库中检索符合条件的向量数据，并支持使用 reranker 进行结果重排序。

## 环境要求
Python 3.7+
torch
modelscope
Milvus Lite
configparser

## 安装
克隆项目到本地：
```bash
git clone https://github.com/your-username/your-repo-name.git
```
安装依赖：

```bash
pip install -r requirements.txt
```

配置 config.ini 文件：
在项目根目录下创建 config.ini 文件，配置 API 密钥：

```bash
ini
[api]
key = your-secret-key
```

## 使用方法

### 向量转换
调用 vector_main 或 vector_list_main 将文本转换为向量。
```python
items = {
    "key": "your-secret-key",
    "text": "输入文本内容"
}
vector = vector_main(items)
```

### 数据插入
调用 milvus_insert_main 将向量数据插入到 Milvus Lite 数据库。
```python
items = {
    "key": "your-secret-key",
    "insert_data": [{"id": 1, "text": "插入的文本内容"}],
    "collection_name": "your-collection-name"
}
milvus_insert_main(items)
```

### 数据查询
调用 milvus_search_main 从 Milvus Lite 数据库中检索符合条件的向量数据，支持使用 reranker 进行结果重排序。
```python
items = {
    "key": "your-secret-key",
    "query": "查询内容",
    "collection_name": "your-collection-name",
    "limit": 5,
    "output_field": ["id", "text"],
    "rerank": 1
}
results = milvus_search_main(items)
print(results)
```

## 许可证
本项目遵循 MIT 许可证。

