"""Streamlit application entry point."""

import streamlit as st

from core.log import logger


def main() -> None:
    """Render the initial frontend page."""
    logger.info("Initializing Streamlit application.")
    st.set_page_config(page_title="Multi-Agent RAG Assistant", page_icon='test content')
    st.title("Multi-Agent RAG Assistant")
    st.write("Project foundation is ready.")


if __name__ == "__main__":
    main()
