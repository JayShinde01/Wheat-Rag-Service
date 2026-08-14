from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

from config import PORT
from rag.chroma_service import ChromaService
from rag.embedding_service import EmbeddingService
from rag.document_service import DocumentService

app = Flask(__name__)
CORS(app)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "https://wheatd.netlify.app",
                "*"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

os.makedirs("uploads", exist_ok=True)

chroma_service = ChromaService()
embedding_service = EmbeddingService()
document_service = DocumentService()


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

        # 1. Extract text

        text = document_service.extract_text(
            file_path
        )

        # 2. Chunk

        chunks = document_service.chunk_text(
            text
        )

        if not chunks:

            return jsonify({
                "error": "No text extracted from PDF"
            }), 400

        # 3. Embeddings

        embeddings = (
            embedding_service.embed_documents(
                chunks
            )
        )

        # 4. IDs

        document_id = str(uuid.uuid4())

        ids = [
            f"{document_id}-{i}"
            for i in range(len(chunks))
        ]

        # 5. Metadata

        metadatas = [
            {
                "document_id": document_id,
                "source": file.filename,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        # 6. Store in ChromaDB

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