import fitz
from rank_bm25 import BM25Okapi

class PDFLibrary:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.chunks = self._load_and_chunk()
        self.bm25 = self._prepare_search()

    def _load_and_chunk(self):
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")
            return []
            
        chunks = []
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip() 
                if len(text) > 40:
                    clean_text = " ".join(text.split())
                    chunks.append(clean_text)
        doc.close()
        return chunks

    def _prepare_search(self):
        if not self.chunks:
            return None
        tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        return BM25Okapi(tokenized_corpus)

    def find_context(self, query, n=1):
        if not self.bm25:
            return "Библиотека пуста."
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.chunks, n=n)
