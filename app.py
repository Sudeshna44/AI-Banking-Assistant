import hashlib
import os

from dotenv import load_dotenv
import streamlit as st
from openai import OpenAIError
from pypdf.errors import PdfReadError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from banking_assistant import (
    ANSWER_NOT_FOUND,
    build_vector_store,
    format_retrieved_context,
    get_source_references,
    make_rag_chains,
)

load_dotenv()


st.set_page_config(page_title="AI Banking Assistant", page_icon="🏦")
st.title("🏦 AI Banking Assistant")
st.caption(
    "Ask questions about the bank documents you upload. Answers are grounded in "
    "those documents; this assistant cannot access personal accounts."
)
st.warning(
    "Do not enter account numbers, card details, passwords, PINs, or other "
    "sensitive personal information."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = ()

with st.sidebar:
    st.header("Bank documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    api_key = st.text_input(
        "OpenRouter API key",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        help="Stored in .env for local use, or enter it for this session only.",
    )

    current_files = tuple(
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in (uploaded_files or [])
    )
    current_file_fingerprint = tuple(
        (name, hashlib.sha256(content).hexdigest())
        for name, content in current_files
    )
    if current_file_fingerprint != st.session_state.indexed_files:
        st.session_state.vector_store = None
        st.session_state.indexed_files = ()

    if st.button("Build / update knowledge base", disabled=not current_files):
        if not api_key.strip():
            st.error("Enter an OpenRouter API key before indexing documents.")
        else:
            try:
                with st.spinner("Reading and indexing your PDF documents..."):
                    embeddings = HuggingFaceEmbeddings(
                        model="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vector_store = build_vector_store(current_files, embeddings)
                st.session_state.vector_store = vector_store
                st.session_state.indexed_files = current_file_fingerprint
                st.session_state.chat_history = []
                st.success(f"Indexed {len(current_files)} PDF document(s).")
            except (ValueError, OSError, PdfReadError, OpenAIError) as exc:
                st.error(f"Could not index the documents: {exc}")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear indexed documents"):
        st.session_state.vector_store = None
        st.session_state.indexed_files = ()
        st.session_state.chat_history = []
        st.rerun()

if not st.session_state.vector_store:
    st.info("Upload bank PDFs and build the knowledge base to get started.")

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input(
    "Ask a question about the uploaded bank documents",
    disabled=st.session_state.vector_store is None,
)

if question:
    if not api_key.strip():
        st.error("Enter an OpenRouter API key in the sidebar to ask a question.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the bank documents..."):
                    llm = ChatOpenAI(
                        model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
                        temperature=0,
                        api_key=api_key.strip(),
                        base_url="https://openrouter.ai/api/v1",
                    )
                    rewrite_chain, answer_chain = make_rag_chains(llm)
                    history = st.session_state.chat_history[:-1]
                    standalone_question = (
                        rewrite_chain.invoke(
                            {"question": question, "chat_history": history}
                        )
                        if history
                        else question
                    )
                    retrieved_docs = st.session_state.vector_store.similarity_search(
                        standalone_question, k=4
                    )
                    answer = answer_chain.invoke(
                        {
                            "question": question,
                            "context": format_retrieved_context(retrieved_docs),
                            "chat_history": history,
                        }
                    ).strip()
                    if not answer:
                        raise RuntimeError("The language model returned an empty answer.")

                st.markdown(answer)
                references = get_source_references(retrieved_docs)
                if references:
                    with st.expander("Sources"):
                        for reference in references:
                            st.markdown(f"- {reference}")
                elif answer == ANSWER_NOT_FOUND:
                    st.caption("No supporting document pages were retrieved.")

                st.session_state.chat_history.append(AIMessage(content=answer))
            except (ValueError, RuntimeError, OSError, OpenAIError) as exc:
                st.session_state.chat_history.pop()
                st.error(f"Unable to answer this question: {exc}")
