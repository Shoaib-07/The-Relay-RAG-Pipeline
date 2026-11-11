# How I Worked with the Assistant — Conversation Record

This file records selected prompts, decisions, debugging steps, and evaluation discussions from the development process. Some early project-planning prompts are summarized rather than reproduced verbatim.

## 1. Three key prompts and why I used them

**Prompt 1 — Retrieval pipeline**

I asked for guidance to build a Python retrieval-augmented pipeline over the three exact arXiv paper versions required by the assignment.

**Why:** To create a traceable evidence retrieval and citation workflow instead of relying on the model's general knowledge.

**Prompt 2 — Local stand-in**

I asked for help implementing a configurable local service that could simulate latency, failures, rate limits, streaming responses, and load behaviour.

**Why:** To test overload and failure scenarios without generating excessive traffic to the real hosted API.

**Prompt 3 — Retrieval and evaluation**

I asked for help comparing 150-, 220-, and 350-word chunks, evaluating TF-IDF retrieval, testing a heuristic reranker, and interpreting load-test results.

**Why:** To choose a retrieval configuration based on measured results and document the limitations.

## 2. Selected question-and-answer tests

The following are actual test questions used with the pipeline:

- "What is multi-head attention, and why does the Transformer use multiple attention heads?"
- "What is the role of the Mimi audio codec in Moshi?"
- "How does H-Net use dynamic chunking to build hierarchical representations of sequences?"
- "What is the exact market price of the H100 GPU in India?"
- "What is the population of India in 2026?"

The first three tested evidence-grounded answers across the required papers. The last two tested whether the pipeline refused unsupported questions.

## 3. One AI mistake I caught

A direct model-only test incorrectly described Mimi as a fictional character in Moshi. I checked the actual paper and then tested the retrieval-grounded pipeline. The grounded pipeline correctly described Mimi as a neural audio codec and cited Moshi passages.

## 4. One assistant recommendation I did not follow

A heuristic reranker was explored, but I did not enable it in the final configuration. Its paper-level retrieval results did not establish reliable passage-level relevance. I selected 350-word chunks, TF-IDF, top five passages, and reranking disabled.

## 5. What was distinctly my work

I configured the Windows development environment, obtained and configured my Groq API key, downloaded and indexed the three papers, executed the real API tests and stand-in load experiments, inspected original PDF pages, checked CSV outputs, compared retrieval configurations, and selected the final system settings.

All reported measurements are based on the experiment outputs saved in the project logs.