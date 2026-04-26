# AI Knowledge Work Assistant

AI Knowledge Work Assistant 是一個面向知識工作場景的研究與問答系統，聚焦於文件解析、混合檢索、多代理協作與即時回應呈現。系統目標是讓使用者可以上傳資料、提出問題，並取得具來源引用的結構化回答。

## Project Status

目前專案處於開發啟動階段，已完成產品需求、系統設計與開發拆解，並已建立 FastAPI / Streamlit 基礎骨架與初始測試。

## Core Capabilities

- 多格式文件匯入與解析，支援 PDF 與 Markdown
- Hybrid Search 檢索流程，結合向量搜尋與關鍵字搜尋
- 多代理工作流，支援研究、整理、產出與審核任務
- 即時串流回應，呈現處理進度與最終答案
- 引用追蹤與 Markdown 呈現，強化答案可驗證性

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- Workflow Orchestration: LangGraph
- LLM Provider: Groq
- Vector Store: FAISS
- Embeddings: sentence-transformers
- Testing: pytest
- Dependency Management: uv
- Deployment: Docker, GitHub Actions, Hugging Face Spaces

## Repository Status

目前 repository 已建立以下工程基礎：

- Git version control
- `.gitignore`
- `.gitattributes`
- `.editorconfig`
- `pyproject.toml`
- `.env.example`
- `app / api / core / tests` 專案骨架
- 初始 pytest 測試

## Planned Structure

```text
app/
api/
core/
tests/
docs/
```

## Quick Start

### Environment

- Python 3.11+
- 已安裝 `uv`

### Local Setup

```bash
uv sync --group dev
```

```bash
copy .env.example .env
```

### Run Backend

```bash
uv run python -m api.main
```

開啟：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### Run Frontend

```bash
uv run streamlit run app/main.py
```

開啟：

```text
http://localhost:8501
```

### Run Tests

```bash
uv run pytest
```

目前測試結構：

```text
tests/unit/         設定與核心邏輯的單元測試
tests/integration/  API 與跨模組整合測試
```

## Development Principles

- 程式碼註解與 docstrings 以繁體中文為主
- 敏感資訊使用環境變數管理，不提交至版本庫
- 每次變更都應附帶對應測試或明確驗證方式
- 優先維持可讀性、可測試性與清楚的模組邊界

更完整的協作與提交流程請參考 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## Roadmap

### v0.1

- 建立專案骨架與依賴管理
- 建立測試框架與品質工具
- 建立設定載入與日誌機制

### v0.2

- 完成文件解析、分塊與索引
- 完成混合檢索與重排序流程

### v0.3

- 完成多代理工作流與外部工具整合
- 完成即時串流 UI 與引用互動

### v1.0

- 完成部署、自動化流程與展示版本

## Architecture & Scalability Notes

目前 MVP 使用 **FastAPI（非同步架構）** 搭配 **In-memory FAISS** 實現 Session 隔離的低延遲向量檢索。

| 層面 | MVP 當前設計 | 若需大規模擴展 |
|---|---|---|
| 向量儲存 | FAISS (In-memory, per-session) | 抽換為 Milvus 叢集 或 Pinecone 雲端服務 |
| Session 持久化 | 記憶體存活期間 | FAISS index 序列化落地硬碟 + TTL 機制 |
| 併發處理 | FastAPI Async IO (單節點) | Kubernetes 水平擴展 + Load Balancer |
| LLM 費用控制 | 使用免費開源模型 (Groq API) | 視需求導入 Semantic Cache 層 |

> **Design Decision**：對於 Side Project Demo 規模（低個位數同時使用者），In-memory FAISS 的回應延遲遠低於外部向量資料庫的網路往返延遲，且完全免除額外基礎設施成本。若未來演進至萬人併發，可在不修改核心 RAG 邏輯的前提下，透過替換 `SessionIndexer` 底層的 index backend 來完成架構升級。
