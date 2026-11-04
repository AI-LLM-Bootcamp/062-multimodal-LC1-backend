# Multimodal PDF App with LangChain 1 · Backend

The backend of a full-stack app that answers questions about PDFs with **text, tables and images**.

You upload a PDF from the frontend. When you ask a question about it, the backend:

1. downloads the PDF from AWS S3,
2. extracts the text and the tables with the **Unstructured API**, and draws each page as a picture with **PyMuPDF**,
3. writes a short summary of every text chunk, every table and every page picture with a vision model (`gpt-6-luna`),
4. stores the summaries in a **multi-vector retriever** (Chroma for the summaries, an in-memory store for the originals),
5. lets a **LangChain 1 agent** search that retriever and answer.

Steps 1 to 4 run only the first time you ask about a PDF. The next questions reuse the same retriever, so they are much faster.

The frontend (Next.js) is in the repository `062-multimodal-LC1-frontend`. The notebooks that explain every change are in `062-multimodal-LC1-notebooks`.

## What you need

- **Python 3.13.11** (we use pyenv to select it)
- **Poetry** to install the project's fixed versions
- **PostgreSQL**, running on your computer
- An **AWS S3 bucket** and its access keys
- An **Unstructured API key** (sign up at [unstructured.io](https://unstructured.io))
- An **OpenAI API key**
- A **LangSmith API key**, to see every step of the app

You do not need Tesseract or Poppler: the Unstructured API does the heavy work on its servers, and PyMuPDF has its own PDF engine.

## Setup

```bash
cd 062-multimodal-LC1-backend
pyenv local 3.13.11
poetry install
```

### Create your .env file

Rename `.env.example` to `.env` and add your own values: database, AWS, Unstructured, OpenAI and LangSmith. Never upload your `.env` file to GitHub.

### Create the database table

Create an empty PostgreSQL database with the name you wrote in `DATABASE_NAME`. Then create the `pdfs` table:

```bash
poetry run alembic upgrade head
```

## Run the backend

```bash
poetry run uvicorn main:app --reload
```

The API runs at http://localhost:8000. Open http://localhost:8000/docs to see and try every endpoint.

## The endpoints

- `POST /pdfs/upload` - upload a PDF file to S3 and save it in the database
- `GET /pdfs` - list the PDFs (add `?selected=true` or `?selected=false` to filter)
- `GET /pdfs/{id}` - get one PDF
- `PUT /pdfs/{id}` - update a PDF
- `DELETE /pdfs/{id}` - delete a PDF
- `POST /pdfs/{id}/ask` - ask a question about a PDF: send `{"question": "..."}`, receive the answer

## Project files

- `main.py` - creates the FastAPI app and the CORS settings
- `routers/pdfs.py` - the endpoints and all the multimodal RAG code
- `config.py` - reads the settings from `.env`
- `database.py`, `models.py`, `schemas.py`, `crud.py` - the database part
- `alembic/` - creates the `pdfs` table
- `pyproject.toml` and `poetry.lock` - the project's fixed versions
- `.env.example` - a model for your `.env` file

## Good to know

- **The first question is slow.** It reads, summarizes and describes the whole PDF. The next questions about the same PDF are fast.
- **The model.** The code uses `gpt-6-luna`, which reads text and images. The name lives in one place, `MODEL = "gpt-6-luna"` at the top of `routers/pdfs.py`, so you can change it easily if OpenAI retires it.
- **Memory.** The retrievers live in the server's memory. If you restart the server, the first question about each PDF builds its retriever again. A real app would save the summaries in a database.
- **Costs.** Each page picture is described once by the vision model, and that uses more tokens than text. Start with short PDFs.
- **Privacy.** Your PDF goes to your S3 bucket, to Unstructured and to OpenAI. Use example documents while you learn, not private ones.
- **PyMuPDF license.** PyMuPDF is licensed under the GNU AGPL 3.0 or a commercial license from Artifex. That is fine for learning; check the license before you use it in a business app.
- **LangSmith.** Open [smith.langchain.com](https://smith.langchain.com) to see every summary, every page description, every search and every answer.
