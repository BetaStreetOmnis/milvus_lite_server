import torch
import torch.nn.functional as F
from modelscope import AutoModelForSequenceClassification, AutoTokenizer

# Initialize the model and tokenizer
model_name_or_path = "iic/gte_passage-ranking_multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True)
model.eval()

{
  "query": "中国的首都在哪儿",
  "documents": [
    "中国的首都在哪儿",
    "北京是中国的首都。",
    "上海是中国的一个主要城市。"
  ]
}
def reranker(query, documents):
    with torch.no_grad():
        # Create pairs of (query, document)
        pairs = [[query, text] for text in documents]
        # Tokenize the input pairs
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        # Pass the tokenized inputs through the model
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        # Convert scores to probabilities in percentage
        probabilities = F.softmax(scores, dim=0) * 100
        
        # Combine text_list with their corresponding probabilities
        ranked_results = list(zip(documents, probabilities.tolist()))
        
        # Sort the results by descending probability
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results
    

# res = reranker("中国的首都在哪儿",[
#     "中国的首都在哪",
#     "北京。",
#     "上海是中国的一个主要城市。"
#   ])




    
