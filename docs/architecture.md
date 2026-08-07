# Detailed Architecture

This is the expanded system view for implementation details, experiments,
and planned extension points. The README keeps a smaller public-facing view.

```mermaid
flowchart TD
    Client(["Client"])

    subgraph API["API Layer — FastAPI"]
        Ingest["POST /ingest"]
        Query["POST /query"]
    end

    Client -->|"documents"| Ingest
    Client -->|"question + filters"| Query

    subgraph IngestPipe["▸ Ingestion Pipeline (implemented)"]
        direction LR
        Loader["Loader<br/>(pdf·docx·html·md)"]
        Cleaner["Cleaner<br/>(text norm)"]
        Chunker["Chunker<br/>(config-driven)"]
        EmbedIngest["Embedder<br/>(sentence-transformers)"]

        Loader --> Cleaner --> Chunker --> EmbedIngest
    end

    Ingest --> Loader
    EmbedIngest -->|"chunks + metadata"| DB[("Postgres + pgvector")]

    subgraph MetaFilter["Metadata Filtering (before retrieval)"]
        direction LR
        F1["✓ dataset_id<br/>(implemented)"]
        F2["✓ category<br/>(implemented)"]
        F3["◐ content_type<br/>(roadmap)"]
        F4["◐ tenant/ACL<br/>(roadmap)"]
    end

    subgraph RetrievePipe["▸ Retrieval Pipeline (implemented)"]
        direction LR
        EmbedQry["Embedder"]
        Dense["Dense Search<br/>(pgvector)"]
        BM25["BM25 Keyword<br/>(rank_bm25)"]
        RRF["RRF Fusion<br/>(hybrid)"]
        Rerank["Reranker<br/>(config-driven)"]

        EmbedQry --> Dense
        EmbedQry -.->|"hybrid mode"| BM25
        Dense --> RRF
        BM25 -.-> RRF
        RRF --> Rerank
        Dense -.->|"dense-only"| Rerank
    end

    Query --> MetaFilter
    MetaFilter --> EmbedQry
    DB --> Dense
    DB -.-> BM25
    Rerank -->|"ranked results"| PromptBuilder

    subgraph GenPipe["▸ Generation Pipeline (implemented)"]
        direction LR
        PromptBuilder["Prompt Builder<br/>(version-aware)"]
        PromptTemplate["📋 Prompt v1/v2<br/>(versioned YAML)"]
        LLM["LLM<br/>(Ollama)"]

        PromptBuilder --> PromptTemplate
        PromptTemplate --> LLM
    end

    LLM -->|"answer + sources"| Query

    subgraph EvalPipe["▸ Evaluation Pipeline (implemented)"]
        direction LR
        Metrics["Metrics Computation"]
        Recall["Recall@5/10<br/>MRR·Hit Rate"]
        RAGAS["RAGAS Scores<br/>(faithfulness)"]

        Metrics --> Recall
        Metrics --> RAGAS
    end

    subgraph Experiment["Experiment Tracking (implemented)"]
        direction LR
        ExpJSON["📊 Experiment JSON<br/>(metrics + config)"]
        Journal["📝 PROJECT_JOURNAL<br/>(session notes)"]
        MLflow["◐ MLflow<br/>(roadmap)"]

        ExpJSON
        Journal
        MLflow
    end

    Rerank -.->|"for eval"| EvalPipe
    LLM -.->|"for eval"| EvalPipe
    Recall --> ExpJSON
    RAGAS --> ExpJSON
    ExpJSON --> Journal

    subgraph Config["🔧 Config-Driven Swaps (config/default.yaml)"]
        direction LR
        SwapChunk["Chunker"]
        SwapEmbed["Embedder"]
        SwapRerank["Reranker"]
        SwapLLM["LLM"]
        SwapPrompt["Prompt Version"]
    end

    Config -.-> Chunker
    Config -.-> EmbedIngest
    Config -.-> Rerank
    Config -.-> LLM
    Config -.-> PromptTemplate

    classDef api fill:#5b8def,stroke:#2f5fc9,color:#fff
    classDef pipeline fill:#3fae5c,stroke:#297a3f,color:#fff
    classDef retrieval fill:#8c5bd6,stroke:#6437a8,color:#fff
    classDef gen fill:#e67e22,stroke:#c66a1a,color:#fff
    classDef eval fill:#27ae60,stroke:#1d7a4a,color:#fff
    classDef storage fill:#e0913b,stroke:#a86420,color:#fff
    classDef config fill:#34495e,stroke:#2c3e50,color:#fff
    classDef roadmap fill:none,stroke:#95a5a6,color:#7f8c8d,stroke-dasharray:5 5
    classDef meta fill:#ecf0f1,stroke:#95a5a6,color:#2c3e50

    class Ingest,Query api
    class Loader,Cleaner,Chunker,EmbedIngest pipeline
    class EmbedQry,Dense,BM25,RRF,Rerank retrieval
    class PromptBuilder,PromptTemplate,LLM gen
    class Metrics,Recall,RAGAS eval
    class DB storage
    class Config config
    class F3,F4,MLflow roadmap
    class MetaFilter meta
```

**Pipeline Flow:**
1. **Ingestion**: Files -> Loader -> Cleaner -> Chunker -> Embedder -> Database
2. **Retrieval**: Question -> Metadata Filter -> Embedder -> Dense/BM25 Search -> RRF Fusion -> Reranker -> Results
3. **Generation**: Results -> Prompt Builder (v1/v2) -> LLM -> Answer
4. **Evaluation**: Retrieval & generation -> Metrics (Recall, MRR, RAGAS) -> Experiment Record -> PROJECT_JOURNAL

**Legend:**
- ✓ = Implemented (solid lines/colors)
- ◐ = Roadmap (dashed borders)
- 🔧 = Config-driven swaps (every colored box is selectable via `config/default.yaml`)
- Dashed arrows = conditional paths (hybrid mode, dense-only mode) or evaluation flow
