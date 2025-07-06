import PyPDF2
import os
from typing import List, Dict
import re

class DocumentProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Извлекает текст из PDF файла"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Ошибка при чтении PDF {file_path}: {e}")
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Извлекает текст из текстового файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Ошибка при чтении TXT {file_path}: {e}")
            return ""
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Разбивает текст на части для лучшей обработки"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def process_reference_documents(self, folder_path: str) -> List[Dict]:
        """Обрабатывает все референсные документы"""
        documents = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if filename.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif filename.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
            else:
                continue
            
            if text:
                chunks = self.split_text_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'id': f"{filename}_{i}",
                        'text': chunk,
                        'source': filename,
                        'chunk_index': i
                    })
        
        return documents