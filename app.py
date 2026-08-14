from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import uuid

from config import PORT
from rag.chroma_service import ChromaService
from rag.embedding_service import EmbeddingService
from rag.document_service import DocumentService


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "https://wheatd.netlify.app"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


os.makedirs("uploads", exist_ok=True)
logger.info("Starting application...")

logger.info("Creating ChromaService...")
chroma_service = ChromaService()
logger.info("ChromaService created")

logger.info("Creating EmbeddingService...")
embedding_service = EmbeddingService()
logger.info("EmbeddingService created")

logger.info("Creating DocumentService...")
document_service = DocumentService()
logger.info("DocumentService created")



@app.get("/")
def health():

    return jsonify({
        "status": "UP",
        "chroma_documents": chroma_service.count()
    })


@app.post("/documents/upload")
def upload_document():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    filename = (
        str(uuid.uuid4())
        + "_"
        + file.filename
    )

    file_path = os.path.join(
        "uploads",
        filename
    )

    file.save(file_path)

    try:

        text = document_service.extract_text(
            file_path
        )

        chunks = document_service.chunk_text(
            text
        )

        if not chunks:

            return jsonify({
                "error": "No text extracted from PDF"
            }), 400

        embeddings = (
            embedding_service.embed_documents(
                chunks
            )
        )

        document_id = str(uuid.uuid4())

        ids = [
            f"{document_id}-{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": document_id,
                "source": file.filename,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        chroma_service.add_documents(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return jsonify({

            "message": "Document processed successfully",

            "documentId": document_id,

            "filename": file.filename,

            "chunks": len(chunks),

            "chromaDocuments":
                chroma_service.count()

        })

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query")
def query():

    data = request.get_json()

    if not data or not data.get("query"):

        return jsonify({
            "error": "query is required"
        }), 400

    question = data["query"]

    query_embedding = (
        embedding_service.embed_query(
            question
        )
    )

    results = chroma_service.search(
        query_embedding,
        top_k=5
    )

    return jsonify({

        "query": question,

        "results": results

    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )