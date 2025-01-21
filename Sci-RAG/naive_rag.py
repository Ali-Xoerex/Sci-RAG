from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class SimpleRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"): # replace with another model
        self.embedding_model = SentenceTransformer(model_name)
        self.qa_pipeline = pipeline(model="deepset/roberta-base-squad2") # replace with another model
        self.chunks = []
        self.embeddings = None

    def load_pdf(self, pdf_path):
        # Extract text from PDF using PyPDF2
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        self.chunk_text(text)

    def chunk_text(self, text, chunk_size=500):
        self.chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.embeddings = self.embedding_model.encode(self.chunks, convert_to_tensor=True)

    def query(self, question):
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, self.embeddings)
        best_chunk_idx = scores.argmax().item()
        best_chunk = self.chunks[best_chunk_idx]
        result = self.qa_pipeline(question=question, context=best_chunk)
        return result["answer"]

rag_system = SimpleRAG()
rag_system.load_pdf("/path/to/pdf/file")
question = "ENTER YOUR QUESTION HERE"
answer = rag_system.query(question)
print(f"Answer: {answer}")