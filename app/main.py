"""Streamlit 應用程式進入點。"""

import streamlit as st

from core.config import settings
from core.log import logger


def main() -> None:
    """渲染初始前端頁面。"""
    logger.info("初始化 Streamlit 應用程式。")
    st.set_page_config(
        page_title="AI Knowledge Work Assistant",
        layout="wide",
    )
    st.title("AI Knowledge Work Assistant")
    st.caption("Story 1.1 foundation: frontend bootstrap is ready.")
    st.write(
        "Backend health endpoint:",
        f"http://{settings.app_host}:{settings.app_port}/health",
    )


if __name__ == "__main__":
    main()
