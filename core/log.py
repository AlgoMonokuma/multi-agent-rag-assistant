"""專案共用日誌模組。"""

import logging
import sys


def setup_logger(name: str = "ai_agent") -> logging.Logger:
    """
    設定並取得標準的 Logger 實體。

    若尚未設定過，會自動加上一個 StreamHandler 以輸出至標準輸出。

    Args:
        name: Logger 的名稱。

    Returns:
        設定好的 logging.Logger 實體。
    """
    logger = logging.getLogger(name)

    # 【工程考量 - 防呆與不可重複性】：
    # 這裡檢查 'not logger.handlers' 是為了避免重複加上「輸出水管(Handler)」。
    # 如果不檢查，每次呼叫這個函式都會多接一根水管，導致同一句話被印出好幾次。
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 【工程考量 - 標準化】：
        # 統一這個專案印出文字的長相。這裡規定了「時間 - 名稱 - 層級 - 訊息」。
        # 比起 print() 只能印出字串，這樣能一眼看出是甚麼時候、在哪裡發生的。
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # StreamHandler 就像是接了一根通往「螢幕標準輸出」的水管
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# 【工程考量 - 模組化與可維護性】：
# 我們在這裡直接產生一個「現成可用」的 logger 實體（名叫 bmad，代表我們這個專案）。
# 其他檔案只需要 `from core.log import logger` 就能直接拿去用，不管在哪個檔案印東西，格式和設定都統一。
logger = setup_logger("bmad")
