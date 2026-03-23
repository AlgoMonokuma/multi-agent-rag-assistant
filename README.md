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

## Engineering Rules

- 程式碼註解與 docstrings 使用繁體中文
- 敏感資訊透過環境變數管理，不提交至版本庫
- 變更需附帶對應測試或驗證方式
- 優先維持可讀性、可測試性與模組邊界清晰

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
