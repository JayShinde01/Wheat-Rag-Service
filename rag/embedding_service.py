from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):
        self.model = None

    def _get_model(self):

        if self.model is None:

            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                device="cpu"
            )

        return self.model

    def embed_documents(self, documents):

        model = self._get_model()

        embeddings = model.encode(
            documents,
            batch_size=4,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, query):

        model = self._get_model()

        embedding = model.encode(
            query,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.tolist()