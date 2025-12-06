
import re
import torch
import pdfplumber
import nltk
from typing import List, Dict
from datasets import Dataset
from transformers import pipeline
from ..utils.config import Config

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

class DataProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract and clean text from a PDF file."""
        print(f"Extracting text from: {pdf_path}")
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        
        # Cleaning
        clean_text = re.sub(r'\n{3,}', '\n\n', full_text)
        clean_text = re.sub(r'Page \d+', '', clean_text)
        print(f"Extracted text length: {len(clean_text)} characters")
        return clean_text

    @staticmethod
    def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks avoiding sentence breaking."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += sent + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        print(f"Created {len(chunks)} chunks")
        return chunks

    @staticmethod
    def generate_qa_dataset(chunks: List[str]) -> Dataset:
        """Generate Q&A pairs from text chunks using T5."""
        print("Initializing Q&A Generation Pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        qg_pipeline = pipeline("text2text-generation", model=Config.QG_MODEL, device=device)
        
        qa_pairs = []
        print("Generating Q&A pairs (this may take time)...")
        
        batch_size = 4 
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            try:
                formatted_inputs = ["<q> " + chunk for chunk in batch_chunks]
                generated_batch = qg_pipeline(formatted_inputs, max_length=512, num_return_sequences=3, do_sample=False)
                
                for j, generated in enumerate(generated_batch):
                    chunk_idx = i + j
                    if chunk_idx >= len(chunks): break
                    original_chunk = chunks[chunk_idx]
                    
                    if isinstance(generated, list):
                        questions = [g['generated_text'].strip() for g in generated]
                    else:
                        questions = [generated['generated_text'].strip()]
                        
                    for question in questions:
                        qa_pairs.append({
                            "instruction": question,
                            "response": original_chunk
                        })
            except Exception as e:
                print(f"Error processing batch starting at chunk {i}: {e}")
                
        qa_pairs = [pair for pair in qa_pairs if len(pair['response']) > 20]
        qa_pairs = list({str(pair): pair for pair in qa_pairs}.values())
        
        dataset = Dataset.from_list(qa_pairs)
        print(f"Created {len(dataset)} Q&A pairs")
        return dataset

    @staticmethod
    def format_instruction(example):
        """Format data for Llama 3 Instruct."""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a legal expert on Royal Saudi Navy jurisdiction and maritime law.<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['response']}<|eot_id|>"""
        return {"text": prompt}
