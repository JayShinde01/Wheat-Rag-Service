from pypdf import PdfReader


class DocumentService:

    def extract_text(self, file_path):

        reader = PdfReader(file_path)

        text_parts = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                text_parts.append(text)

        return "\n".join(text_parts)

    def chunk_text(
        self,
        text,
        chunk_size=1000,
        overlap=200
    ):

        chunks = []

        step = chunk_size - overlap

        for start in range(0, len(text), step):

            chunk = text[start:start + chunk_size].strip()

            if chunk:
                chunks.append(chunk)

        return chunks