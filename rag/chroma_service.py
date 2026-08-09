import chromadb

from config import CHROMA_URL, COLLECTION_NAME


class ChromaService:

    def __init__(self):

        self.client = chromadb.HttpClient(
            host=CHROMA_URL.replace("https://", "").replace("http://", ""),
            port=443,
            ssl=CHROMA_URL.startswith("https://")
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def add_documents(
            self,
            ids,
            documents,
            embeddings,
            metadatas):

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
            self,
            query_embedding,
            top_k=5):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def count(self):

        return self.collection.count()