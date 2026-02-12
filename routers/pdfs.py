from typing import List
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
import schemas
import crud
from database import SessionLocal
from uuid import uuid4

# LangChain 1.0 imports for Multimodal RAG
import tempfile
import os
import uuid
import base64
import requests
import fitz  # pymupdf: renders PDF pages as images
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.schema.document import Document
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from unstructured_client import UnstructuredClient
from config import get_settings

router = APIRouter(prefix="/pdfs")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= EXISTING ENDPOINTS (unchanged) =============

@router.post("", response_model=schemas.PDFResponse, status_code=status.HTTP_201_CREATED)
def create_pdf(pdf: schemas.PDFRequest, db: Session = Depends(get_db)):
    return crud.create_pdf(db, pdf)

@router.post("/upload", response_model=schemas.PDFResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_name = f"{uuid4()}-{file.filename}"
    return crud.upload_pdf(db, file, file_name)

@router.get("", response_model=List[schemas.PDFResponse])
def get_pdfs(selected: bool = None, db: Session = Depends(get_db)):
    return crud.read_pdfs(db, selected)

@router.get("/{id}", response_model=schemas.PDFResponse)
def get_pdf_by_id(id: int, db: Session = Depends(get_db)):
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return pdf

@router.put("/{id}", response_model=schemas.PDFResponse)
def update_pdf(id: int, pdf: schemas.PDFRequest, db: Session = Depends(get_db)):
    updated_pdf = crud.update_pdf(db, id, pdf)
    if updated_pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return updated_pdf

@router.delete("/{id}", status_code=status.HTTP_200_OK)
def delete_pdf(id: int, db: Session = Depends(get_db)):
    if not crud.delete_pdf(db, id):
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"message": "PDF successfully deleted"}

# ============= MULTIMODAL RAG FUNCTIONALITY =============

def download_pdf_from_url(url: str) -> str:
    """Download a PDF from a URL and save it to a temporary file."""
    response = requests.get(url)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()

    return temp_file.name


# ========== PDF EXTRACTION (replaces PyPDFLoader) ==========

def extract_pdf_elements(pdf_path: str):
    """
    Extract text, tables, and images from a PDF file.

    - Text and tables: extracted via the Unstructured API (good at structured content)
    - Page images: rendered via pymupdf and sent to GPT-4o for description

    pymupdf renders each page as an image, which guarantees GPT-4o sees everything
    on the page — including vector-based charts that the Unstructured API may miss.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        tuple: (text_elements, table_elements, image_base64_list)
    """
    settings = get_settings()
    client = UnstructuredClient(api_key_auth=settings.UNSTRUCTURED_API_KEY)

    print(f"Processing PDF via Unstructured API: {pdf_path}")

    with open(pdf_path, "rb") as f:
        file_content = f.read()

    # Use Unstructured API for text and table extraction
    response = client.general.partition(
        request={
            "partition_parameters": {
                "files": {
                    "content": file_content,
                    "file_name": os.path.basename(pdf_path),
                },
                "strategy": "hi_res",
                "infer_table_structure": True,
            }
        }
    )

    text_types = {
        "NarrativeText", "Title", "UncategorizedText",
        "ListItem", "Header", "Footer", "FigureCaption",
    }

    raw_texts = []
    table_elements = []

    for element in response.elements:
        el_type = element.get("type", "")
        text = element.get("text", "")

        if el_type in text_types and text.strip():
            raw_texts.append(text)
        elif el_type == "Table":
            table_elements.append(text)

    # Group small text elements into ~2000-char chunks
    text_elements = []
    current_chunk = ""
    for text in raw_texts:
        if len(current_chunk) + len(text) > 2000 and current_chunk:
            text_elements.append(current_chunk.strip())
            current_chunk = text
        else:
            current_chunk += "\n\n" + text if current_chunk else text
    if current_chunk.strip():
        text_elements.append(current_chunk.strip())

    # Render each PDF page as an image using pymupdf
    # This captures everything visible on the page: charts, diagrams, photos, etc.
    image_base64_list = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_base64_list.append(img_b64)
        print(f"  Rendered page {page_num} as image")
    doc.close()

    print(f"Extracted:")
    print(f"  - {len(text_elements)} text elements (from {len(raw_texts)} raw)")
    print(f"  - {len(table_elements)} table elements")
    print(f"  - {len(image_base64_list)} page images")

    return text_elements, table_elements, image_base64_list


# ========== SUMMARIZATION FUNCTIONS ==========

def summarize_image(image_base64: str, model: ChatOpenAI) -> str:
    """
    Use GPT-4o to describe an image.

    Args:
        image_base64: Base64-encoded image string (from Unstructured API)
        model: ChatOpenAI model instance with vision capabilities (gpt-4o or gpt-4o-mini)

    Returns:
        str: Description of the image
    """
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Describe this image in detail. Include:
                - What type of content it shows (chart, diagram, photo, etc.)
                - Any text visible in the image
                - Key data points if it's a chart or graph
                - The overall meaning or purpose of the image"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]
    )

    response = model.invoke([message])
    return response.content


def summarize_text(text: str, model: ChatOpenAI) -> str:
    """Summarize a text element."""
    prompt = f"""Summarize the following text concisely while preserving key information:

{text}

Summary:"""
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


def summarize_table(table: str, model: ChatOpenAI) -> str:
    """Summarize a table element."""
    prompt = f"""Summarize the following table, highlighting key data points and relationships:

{table}

Summary:"""
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


def create_all_summaries(text_elements, table_elements, image_base64_list):
    """
    Create summaries for all content types.

    Uses gpt-4o-mini for everything — text, tables, and images.
    It's fast, cheap, multimodal, and recommended over the legacy gpt-3.5-turbo.

    Returns:
        tuple: (text_summaries, table_summaries, image_summaries)
    """
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    print("Summarizing text elements...")
    text_summaries = []
    for i, text in enumerate(text_elements, 1):
        try:
            summary = summarize_text(text, model)
            text_summaries.append(summary)
            print(f"  Text {i}/{len(text_elements)} done")
        except Exception as e:
            print(f"  Error on text {i}: {e}")
            text_summaries.append(f"Error: {str(e)[:100]}")

    print("Summarizing table elements...")
    table_summaries = []
    for i, table in enumerate(table_elements, 1):
        try:
            summary = summarize_table(table, model)
            table_summaries.append(summary)
            print(f"  Table {i}/{len(table_elements)} done")
        except Exception as e:
            print(f"  Error on table {i}: {e}")
            table_summaries.append(f"Error: {str(e)[:100]}")

    print("Summarizing page images with GPT-4o-mini...")
    image_summaries = []
    for i, image_b64 in enumerate(image_base64_list, 1):
        try:
            summary = summarize_image(image_b64, model)
            image_summaries.append(summary)
            print(f"  Image {i}/{len(image_base64_list)} done")
        except Exception as e:
            print(f"  Error on image {i}: {e}")
            image_summaries.append(f"Error: {str(e)[:100]}")

    return text_summaries, table_summaries, image_summaries


# ========== MULTI-VECTOR RETRIEVER ==========

def create_multimodal_retriever(
    text_summaries, text_elements,
    table_summaries, table_elements,
    image_summaries
):
    """
    Create a Multi-Vector Retriever for multimodal content.

    The Multi-Vector Retriever pattern:
    - Vector Store: Contains embeddings of summaries (for fast semantic search)
    - Doc Store: Contains original content (for accurate answers)
    - UUIDs link summaries to their original content

    Returns:
        MultiVectorRetriever: Configured retriever
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name="multimodal_summaries",
        embedding_function=embeddings
    )

    docstore = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key
    )

    def add_documents(summaries, original_contents, content_type):
        """Helper to add documents with proper UUIDs."""
        if not summaries:
            return

        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [
            Document(
                page_content=s,
                metadata={id_key: doc_ids[i], "type": content_type}
            )
            for i, s in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_contents)))
        print(f"  Added {len(summaries)} {content_type} documents")

    print("Building vector store...")
    add_documents(text_summaries, text_elements, "text")
    add_documents(table_summaries, table_elements, "table")
    # For images, we store the summary as both summary AND original
    # (since the base64 image isn't useful for text answers)
    add_documents(image_summaries, image_summaries, "image")

    print("Vector store ready!")
    return retriever


# ========== MULTIMODAL RAG AGENT ==========

def create_multimodal_rag_agent(retriever):
    """
    Create a LangChain 1.0 RAG agent with multimodal search capabilities.

    Uses the modern create_agent function introduced in LangChain 1.0.
    The agent decides when to use tools and can reason about queries.

    Args:
        retriever: The MultiVectorRetriever with our document content

    Returns:
        agent: A configured LangChain agent
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Search the document database for relevant information.
        Use this tool when you need to find specific information from the documents.
        The database contains text, tables, and image descriptions from PDF documents.
        """
        results = retriever.invoke(query)

        if not results:
            return "No relevant information found in the documents."

        formatted_results = []
        for i, result in enumerate(results, 1):
            if hasattr(result, 'page_content'):
                formatted_results.append(f"[Result {i}]\n{result.page_content}")
            else:
                formatted_results.append(f"[Result {i}]\n{result}")

        return "\n\n".join(formatted_results)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_documents],
        system_prompt="""You are a helpful assistant that answers questions about documents.

You have access to a document database that contains:
- Text content from PDF documents
- Table data (financial figures, statistics, etc.)
- Descriptions of images (charts, diagrams, photos)

When answering questions:
1. Use the search_documents tool to find relevant information
2. Base your answers ONLY on the retrieved information
3. If the information isn't in the documents, say so clearly
4. For numerical questions, quote the exact figures from the documents
5. For image-related questions, refer to the image descriptions

Be concise but accurate."""
    )

    return agent


# ========== ASK ENDPOINT (Multimodal) ==========

@router.post("/{id}/ask")
async def ask_pdf(id: int, request: schemas.QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question about a specific PDF document using Multimodal RAG."""

    # Get the PDF from database
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")

    temp_path = None
    try:
        # Step 1: Download PDF from S3
        temp_path = download_pdf_from_url(pdf.file)

        # Step 2: Extract text, tables, AND images via Unstructured API
        text_elements, table_elements, image_base64_list = extract_pdf_elements(
            temp_path
        )

        # Step 3: Create summaries for all content types
        text_summaries, table_summaries, image_summaries = create_all_summaries(
            text_elements, table_elements, image_base64_list
        )

        # Step 4: Build the Multi-Vector Retriever
        retriever = create_multimodal_retriever(
            text_summaries, text_elements,
            table_summaries, table_elements,
            image_summaries
        )

        # Step 5: Create the multimodal RAG agent
        agent = create_multimodal_rag_agent(retriever)

        # Step 6: Ask the question
        response = agent.invoke(
            {"messages": [HumanMessage(content=request.question)]}
        )

        # Extract the answer
        answer = response["messages"][-1].content

        return {
            "pdf_id": id,
            "pdf_name": pdf.name,
            "question": request.question,
            "answer": answer
        }

    finally:
        # Clean up temporary PDF file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
