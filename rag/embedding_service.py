from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    def embed_documents(self, documents):

        embeddings = self.model.encode(
            documents,
            batch_size=4,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, query):

        embedding = self.model.encode(
            query,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.tolist()