# 貢獻指南 (Contributing Guide)

感謝您參與 **AI Knowledge Work Assistant** 的開發！為了確保程式碼品質與團隊協作順暢，請所有貢獻者嚴格遵守以下開發與文件規範。

## 目錄
1. [開發環境設定](#開發環境設定)
2. [語言與註解規範 (CRITICAL)](#語言與註解規範-critical)
3. [日誌記錄規範](#日誌記錄規範)
4. [測試與程式碼審查](#測試與程式碼審查)

---

## 開發環境設定

本專案使用 `uv` 進行依賴管理，並且強制要求設定安全環境變數。

1. **安裝依賴**: 
   請在專案根目錄下執行：
   ```bash
   uv sync
   ```
2. **環境變數設定**: 
   請複製 `.env.example` 為 `.env`，並補上您的 API Keys (如 `GROQ_API_KEY`)。
   > ⚠️ 絕對不要將含有真實金鑰的 `.env` 檔案 commit 到版本控制系統中。

---

## 語言與註解規範 (CRITICAL)

為了符合專案在地化需求，本專案的註解語言受到**嚴格限制**：

- **繁體中文唯一原則**: 所有的 code comments、Docstrings (模組、類別、函式的說明) **必須完全使用繁體中文** 撰寫！
- 禁止使用英文或簡體中文做為主要的文件語言。
- 變數名稱與函數名稱請保持 `snake_case` 或 `PascalCase` 英文命名。

### 範例：
```python
def process_data(data_id: str) -> bool:
    """
    處理傳入的資料 ID 並回傳是否成功。

    Args:
        data_id: 要處理的資料唯一識別碼。

    Returns:
        處理成功回傳 True，否則為 False。
    """
    # 執行資料清理
    clean_id = data_id.strip()
    return True
```

---

## 日誌記錄規範

為了保持除錯時的資訊清晰，並與雲端部署順利整合，請遵守以下日誌規範：

1. **一律使用 `core.log.logger`**: 請勿使用內建的 `print()`。
2. **嚴謹的記錄等級 (Log Level)**:
   - `INFO`: 重要業務流程啟動、成功完成。
   - `ERROR`: 系統發生異常，需要被監控並捕捉到的例外情形。
   - `DEBUG`: (可選) 僅用於本機除錯，請勿在生產環境留下太多 debug logs。

### 使用方法：
```python
from core.log import logger

logger.info("系統初始化成功。")
try:
    ...
except Exception as e:
    logger.error(f"發生未預期的錯誤: {e}")
```

---

## 測試與程式碼審查

- 本專案採用 **測試驅動開發 (TDD)**。在提交新功能前，請確保對應的單元測試位在 `/tests/unit` 目錄下。
- 使用 pytest 進行測試：
  ```bash
  pytest
  ```
- 提交 PR 前，請務必跑過一次所有測試，並確保沒有破壞現有功能。
