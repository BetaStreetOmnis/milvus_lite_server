## Introduction
This project leverages Milvus Lite for vector search and reranking. It implements functionalities to convert text into vectors, insert vector data, retrieve data from Milvus Lite, and rerank the results using a custom reranker. The project is implemented in Python and provides operations via a RESTful API.

## Features
Vector Conversion: Convert input text into vector representations.
Data Insertion: Insert vectorized text data into the Milvus Lite database.
Data Retrieval: Retrieve vector data from the Milvus Lite database based on search criteria, with support for reranking the results.

## Requirements
Python 3.7+
torch
modelscope
Milvus Lite
configparser

## Installation
Clone the repository:
`git clone https://github.com/BetaStreetOmnis/milvus_lite_server`

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure config.ini file: Create a config.ini file in the project's root directory to set up your API key:
```bash
[api]
key = your-secret-key
```

## Usage

### Vector Conversion
To convert text into vectors, call vector_main or vector_list_main:
```python
items = {
    "key": "your-secret-key",
    "text": "Input text content"
}
vector = vector_main(items)
```
### Data Insertion
To insert vector data into the Milvus Lite database, use milvus_insert_main:
```python
items = {
    "key": "your-secret-key",
    "insert_data": [{"id": 1, "text": "Text content to insert"}],
    "collection_name": "your-collection-name"
}
milvus_insert_main(items)
```

### Data Retrieval
To retrieve vector data from the Milvus Lite database and optionally rerank the results, call milvus_search_main:
```python
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
```

## License
This project is licensed under the MIT License.

