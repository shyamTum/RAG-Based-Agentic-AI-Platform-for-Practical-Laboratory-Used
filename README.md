# RAG-Based Agentic AI Platform for Domain-Specific Knowledge Retrieval

An agent-enhanced Retrieval-Augmented Generation (RAG) platform designed for enterprise document intelligence and domain-specific knowledge retrieval.

This project extends a traditional RAG pipeline with lightweight agentic reasoning capabilities using LangGraph. The system performs query planning, retrieval validation, retrieval expansion, and transparent execution tracing before generating a final answer.

---

## Features

- Semantic document retrieval using SentenceTransformers and FAISS
- Multi-LLM response generation
- Source-aware answer generation
- Conversational memory support
- Query rewriting for follow-up questions
- LangGraph-based agentic workflow
- Retrieval quality validation
- Automatic retrieval expansion when evidence is insufficient
- Transparent retrieval trace and statistics
- Interactive Gradio interface

---

## Baseline RAG Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/d5b59b1d-1ac9-4073-84f0-9d21caaf1316" width="450">
</p>

<p align="center">
<i>Baseline RAG Architecture</i>
</p>

### Workflow

1. Documents are converted into embeddings.
2. Embeddings are stored in a FAISS vector database.
3. User queries are embedded into the same vector space.
4. Top-k relevant document chunks are retrieved.
5. Retrieved chunks are combined into context.
6. The context is sent to an LLM.
7. The system returns the answer together with source references.

---

## Conversational RAG Extension

The baseline workflow was extended with conversational memory to support follow-up questions and multi-turn interactions.

### Key Enhancements

- Conversation history maintained across interactions
- Follow-up questions rewritten into standalone queries
- Rewritten query used for document retrieval
- Retrieved context combined with conversation history

### Example

**User Question**

```text
What is Grounded DINO?
```

**Follow-up Question**

```text
What are its limitations?
```

**Rewritten Query**

```text
What are the limitations of Grounded DINO?
```

This allows the retriever to access the correct document context even when follow-up questions contain ambiguous references such as "it", "they", or "those methods".

---

## Lightweight Agentic RAG Extension

The conversational RAG workflow was further extended using LangGraph to introduce lightweight agentic retrieval intelligence.

<p align="center">
  <img width="250" height="360" alt="image" src="https://github.com/user-attachments/assets/e107950c-0025-4d00-9179-83d5f33534d4" />
</p>

<p align="center">
<i>Lightweight Agentic RAG Workflow</i>
</p>

### Agentic Workflow

1. Query Planner generates retrieval-focused search queries.
2. Retriever gathers candidate document chunks.
3. Retrieval Quality Check evaluates retrieval effectiveness.
4. If evidence is weak, Expanded Retrieval is triggered.
5. Retrieved evidence is passed to the LLM.
6. Final answer is generated with source references.

### Retrieval Quality Validation

The system evaluates retrieval quality using:

- FAISS similarity scores
- Number of unique retrieved chunks
- Retrieval coverage statistics

When retrieval quality is below configured thresholds, additional retrieval is automatically performed before answer generation.

### Why Agentic Retrieval?

Compared with a traditional RAG system, the agentic workflow provides:

- Better retrieval coverage
- Improved evidence collection
- Reduced risk of incomplete context
- More transparent retrieval behavior
- Explainable execution traces

---

## LangGraph State Flow

The LangGraph workflow maintains a shared state that is passed between nodes.

Typical state information includes:

```text
User Question
Rewritten Question
Planned Retrieval Queries
Retrieved Chunks
Retrieval Scores
Execution Trace
Generated Answer
```

Each node updates the state and passes it to the next node in the workflow.

---

## Technology Stack

### Retrieval

- FAISS
- SentenceTransformers
- Vector Similarity Search

### Agentic Layer

- LangGraph
- LangChain

### Backend

- Python
- Pandas
- NumPy

### Interface

- Gradio

### LLM Integration

- Enterprise LLM APIs
- Prompt Engineering
- Context-Aware Retrieval

---

## Application Screenshots

### Agentic RAG Interface

<p align="center">
  <img src="YOUR_APP_SCREENSHOT_1" width="900">
</p>

### Retrieval Trace and Agent Workflow

<p align="center">
  <img src="YOUR_APP_SCREENSHOT_2" width="900">
</p>

The interface displays:

- Generated answer
- Source references
- Retrieval statistics
- Agent execution trace
- Retrieval quality metrics

---

## Repository Structure

```text
.
├── app_gradio_with_agentic_features.py
├── ingest_kb.py
├── loader.py
├── embeddings/
├── faiss_index/
├── data/
└── README.md
```

---

## Future Scope: Knowledge Base Enhancement
<p align="center">
  <img width="857" height="168" alt="image" src="https://github.com/user-attachments/assets/810f8748-4b24-4a76-852f-4fd0fa28ee2f" />
</p>

Future improvements include:

- Semantic chunking
- Overlap-aware chunking
- Rich metadata generation
- Document summarization
- Keyword extraction
- Hybrid vector and relational storage

Potential metadata fields:

- File type
- Page number
- Section title
- Keywords
- Source path
- Document summary

---

## Future Scope: Agentic Intelligence Layer

<p align="center">
  <img width="812" height="252" alt="image" src="https://github.com/user-attachments/assets/35557833-8d4a-48e4-a637-f3815d32dd92" />
</p>

Potential future extensions include:

### Reasoning Layer

- Planner Agent
- Executor Agent
- Conditional Router
- Verifier Agent

### Multi-Agent System

Specialized agents for:

- Imaging workflows
- Gene expression analysis
- Color science
- General document QA

### Validation System

- Human validation
- LLM-based validation
- Retrieval verification
- Grounding verification
- Latency monitoring
- Cost monitoring

---

## Disclaimer

This repository demonstrates an agent-enhanced document intelligence platform for research and engineering purposes.

No proprietary datasets, confidential enterprise documents, internal infrastructure details, or organization-specific information are included in this repository.
