from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def embed_documents(self, documents):

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def embed_query(self, query):

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding.tolist()