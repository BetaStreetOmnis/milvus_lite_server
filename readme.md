## Introduction
This is a project that uses Milvus Lite for vector search and reranking. The project implements functionalities to convert text into vectors, insert vector data, retrieve data from Milvus Lite, and rerank the results using a reranker. The project is implemented in Python and supports operations via RESTful API.

## Features
Vector Conversion: Convert input text into vector representations.
Data Insertion: Insert vectorized text data into Milvus Lite database.
Data Retrieval: Retrieve vector data from Milvus Lite database based on criteria, with support for reranking the results.
Requirements
Python 3.7+
torch
modelscope
Milvus Lite
configparser
Installation
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
Install dependencies:

pip install -r requirements.txt
Configure config.ini file:
Create a config.ini file in the project root directory to configure the API key:


[api]
key = your-secret-key
Usage
Vector Conversion

Call vector_main or vector_list_main to convert text into vectors.


items = {
    "key": "your-secret-key",
    "text": "Input text content"
}
vector = vector_main(items)
Data Insertion

Call milvus_insert_main to insert vector data into Milvus Lite database.


items = {
    "key": "your-secret-key",
    "insert_data": [{"id": 1, "text": "Text content to insert"}],
    "collection_name": "your-collection-name"
}
milvus_insert_main(items)
Data Retrieval

Call milvus_search_main to retrieve vector data from Milvus Lite database based on criteria, with support for reranking the results.


items = {
    "key": "your-secret-key",
    "query": "Search query",
    "collection_name": "your-collection-name",
    "limit": 5,
    "output_field": ["id", "text"],
    "rerank": 1
}
results = milvus_search_main(items)
print(results)
License
This project is licensed under the MIT License.