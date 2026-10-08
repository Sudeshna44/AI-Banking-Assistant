# AI Banking Assistant

A Streamlit training app that answers questions using only the bank PDF documents
uploaded by the user. It demonstrates PDF loading, text chunking, local embeddings,
Chroma retrieval, LangChain prompts, conversational follow-ups, and source citations.

## Requirements

- Python 3.10 or newer
- An OpenRouter API key for chat completions
- Internet access on the first run to download the local embedding model
- The packages in `requirements.txt`

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Set `OPENROUTER_API_KEY` in `.env` to your OpenRouter key. The included `.env`
contains a placeholder; replace it with your key before starting the app. Do not
share or commit this file. The chat model defaults to `openrouter/free`; set
`OPENROUTER_MODEL` in `.env` to select another model available to your account.
You can also enter the key in the app sidebar for that session.

Chat completions use OpenRouter's OpenAI-compatible API. Text embeddings run
locally with `sentence-transformers/all-MiniLM-L6-v2`, so PDF indexing does not
require an OpenRouter embeddings model or send embedding requests to OpenRouter.
The first installation needs the `sentence-transformers` package and internet
access to download the embedding model.

Sample fictional PDFs for testing are in [sample_documents](./sample_documents/).
They are not real bank policies, rates, eligibility rules, or customer advice.

## Use

1. Upload one or more bank policy/FAQ PDFs in the sidebar.
2. Select **Build / update knowledge base**.
3. Ask a question in the chat. Follow-up questions can refer to earlier turns.
4. Open **Sources** under an answer to see the retrieved PDF pages.

Changing the uploaded files invalidates the current index; build it again to use
the new documents. Indexes and conversation history are kept in Streamlit session
memory and are not persisted to disk. The index is rebuilt when the app session
ends.

## Safety and scope

- Answers are instructed to use only retrieved document context and to say when
  that context does not contain the answer. This is a best-effort grounding
  approach, not a guarantee of accuracy.
- The app cannot see account records and must not be used for account access,
  identity verification, or personalized lending decisions.
- Do not upload documents containing customer records or enter account numbers,
  card details, credentials, or other sensitive personal data.
- PDF text is sent to OpenRouter for chat answers; embeddings are generated
  locally. Use only documents and services approved by your organization.

This example is for training and is not a production banking system.
