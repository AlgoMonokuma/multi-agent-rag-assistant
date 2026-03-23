"""Streamlit application entrypoint."""

import streamlit as st

from core.config import settings


def main() -> None:
    """Render the bootstrap frontend page."""
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
