# Fashion Multimodal Retrieval (CLIP + Attribute Decomposition)

This project implements a multimodal fashion retrieval system using CLIP with
explicit attribute decomposition to handle compositional and context-aware queries.

## Attributes Modeled
- Clothing type
- Color
- Context (environment)
- Style (formal/casual)

## Why Attribute Decomposition?
Vanilla CLIP struggles with compositional queries (e.g. red shirt + blue pants).
We encode and index each attribute separately and fuse similarity scores at query time.

## Pipeline
1. Offline image indexing (attribute-wise embeddings)
2. Query decomposition
3. Attribute-wise retrieval
4. Weighted score fusion

## Run
```bash
bash scripts/run_indexer.sh
bash scripts/run_retriever.sh
