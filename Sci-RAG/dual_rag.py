import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoder, T5ForConditionalGeneration, T5Tokenizer 


class DualRAG:
    def __init__(self, question_model="facebook/dpr-question_encoder-single-nq-base",
                 context_encoder="facebook/dpr-ctx_encoder-single-nq-base",
                 tokenizer_model="t5-small",
                 generator_model="t5-small"): # Experiment with different models for different parts
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_model)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder)
        self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_model)
        self.indexer = None
        self.texts = None

    def load_knowledge(self, knowledge_path):
        # Extract text from PDF using PyPDF2
        pdf_path = filter(lambda x: x.endswith(".pdf"),os.listdir(knowledge_path))
        reader = [PdfReader(pdf) for pdf in pdf_path]
        self.texts = [" ".join([page.extract_text() for page in r.pages if page.extract_text()]) for r in reader]
        # tokenize the context
        document_embeddings = []
        for doc in self.texts:
            # Tokenize the document
            inputs = self.context_tokenizer(doc, return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids
            # Encode the document
            embeddings = self.context_encoder(inputs)["pooler_output"].detach().numpy()
            document_embeddings.append(embeddings)
        document_embeddings = np.vstack(document_embeddings)
        # build an index for efficient knowledge storing
        d = document_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(document_embeddings)
        self.indexer = index

    def query(self, question):
        question_inputs = self.question_tokenizer(question,return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids
        question_embedding = self.question_encoder(question_inputs)["pooler_output"].detach().numpy()
        k = 2 # number of documents to retrieve
        distances,indices = self.indexer.search(question_embedding,k)
        retrieved_documents = [self.texts[i] for i in indices[0]]
        # Now we shall generate a response with a generator model
        input_text = f"question: {question} context: {' '.join(retrieved_documents)}"
        input_ids = self.tokenizer(input_text,return_tensors="pt",padding=True,truncation=True,max_length=512).input_ids
        generated_ids = self.generator.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        response = self.tokenizer.batch_decode(generated_ids[0],skip_special_tokens=True)
        return response

rag_system = DualRAG()
rag_system.load_knowledge("/path/to/knowledge/base")
question = "WRITE YOUR QUERY HERE"
answer = rag_system.query(question)
print("Answer: "+" ".join(answer))
