# AI Knowledge Work Assistant

一個針對知識工作場景設計的文件問答系統。使用者上傳文件後，系統透過混合檢索（向量 + 關鍵字）找出相關段落，再交由 LLM 生成具來源引用的結構化回答。

## Core Capabilities

- PDF 與 Markdown 文件解析
- 以 FAISS 為底層的 Session 隔離向量索引
- Hybrid Search：向量搜尋（sentence-transformers）+ BM25 關鍵字搜尋
- 多代理工作流（LangGraph）：研究、整理、產出、審核
- 即時串流回應（SSE）與引用來源追蹤

## Tech Stack

| 類別 | 技術 |
|---|---|
| Backend | FastAPI + Async IO |
| Frontend | Streamlit |
| Workflow | LangGraph |
| LLM | Groq API |
| Vector Store | FAISS (In-memory, per-session) |
| Embeddings | sentence-transformers |
| Testing | pytest |
| Dependency | uv |
| Deployment | Docker, GitHub Actions, Hugging Face Spaces |

## Project Structure

```text
core/rag/
  parser.py       ✅ PDF / Markdown 解析
  chunker.py      ✅ RecursiveCharacterTextSplitter 分塊
  embeddings.py   ✅ sentence-transformers 向量化
  indexer.py      ✅ Session 隔離 FAISS 索引管理
  retriever.py    ✅ Hybrid Search (向量 + BM25)
  pipeline.py     ✅ 文件攝入管線

app/              前端 Streamlit（開發中）
api/              FastAPI 路由（開發中）
tests/            49 個單元與整合測試全數通過
```

## Quick Start

```bash
# 安裝依賴
uv sync --group dev

# 複製環境設定
copy .env.example .env
```

```bash
# 啟動後端
uv run python -m api.main
# http://127.0.0.1:8000/health

# 啟動前端
uv run streamlit run app/main.py
# http://localhost:8501

# 執行測試
uv run pytest
```

## Architecture Notes

系統採用 **In-memory FAISS per-session** 的設計：每個使用者對話建立獨立的 FAISS 索引，避免不同使用者的文件語意互相污染，並在對話生命週期結束後自動釋放記憶體。

若需支援更大規模的部署：

| 層面 | 當前設計 | 可能的演進方向 |
|---|---|---|
| 向量儲存 | FAISS In-memory | Milvus 叢集 / Pinecone |
| Session 持久化 | 記憶體存活期間 | FAISS index 序列化 + TTL |
| 水平擴展 | 單節點 Async IO | Kubernetes + Load Balancer |

## Development Principles

- 程式碼註解與 docstrings 以繁體中文撰寫
- 敏感資訊以環境變數管理，不提交至版本庫
- 每次功能變更須附帶對應單元測試
- 模組邊界清楚，RAG 邏輯不散落至 API 或 UI 層

詳見 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## Roadmap

| 版本 | 目標 | 狀態 |
|---|---|---|
| v0.1 | 專案骨架、測試框架、設定與日誌 | ✅ 完成 |
| v0.2 | 文件解析、分塊、索引、Hybrid Search、重排序 | 🔄 進行中 |
| v0.3 | 多代理工作流、外部工具、串流 UI | 未開始 |
| v1.0 | 部署、CI/CD、展示版本 | 未開始 |

