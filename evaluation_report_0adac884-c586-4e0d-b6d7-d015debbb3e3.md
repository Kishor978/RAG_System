# RAG System Evaluation Report
Date: 2025-08-01 16:01:10

## Summary
Best Configuration: **fixed_size** chunking with **cosine** similarity
Best F1 Score: **0.6207**
Notes: Evaluation completed on 12 test queries against 10 documents.

## Detailed Results
| Chunking Method | Similarity Algorithm | Accuracy | Precision | Recall | F1 Score | Latency (ms) |
|----------------|----------------------|----------|-----------|--------|----------|--------------|
| fixed_size | cosine | 0.4500 | 0.4500 | 1.0000 | 0.6207 | 56.01 |
| fixed_size | dot_product | 0.4500 | 0.4500 | 1.0000 | 0.6207 | 62.57 |
| recursive_character | cosine | 0.4500 | 0.4500 | 1.0000 | 0.6207 | 33.60 |
| recursive_character | dot_product | 0.4500 | 0.4500 | 1.0000 | 0.6207 | 44.87 |

## Analysis
### Chunking Methods Comparison
- **fixed_size**: Average F1: 0.6207, Average Latency: 59.29 ms
- **recursive_character**: Average F1: 0.6207, Average Latency: 39.24 ms

### Similarity Algorithms Comparison
- **dot_product**: Average F1: 0.6207, Average Latency: 53.72 ms
- **cosine**: Average F1: 0.6207, Average Latency: 44.81 ms