"""
PDF loading and text extraction module
"""
from pypdf import PdfReader
from typing import List, Dict
import os


class PDFLoader:
    """PDF document loader"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.validate_file()
    
    def validate_file(self):
        """Validate PDF file exists and is readable"""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        if not self.pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {self.pdf_path}")
        
        if not os.access(self.pdf_path, os.R_OK):
            raise PermissionError(f"Cannot read PDF file: {self.pdf_path}")
    
    def load(self) -> str:
        """
        Load and extract text from PDF
        
        Returns:
            Extracted text as a single string
        """
        try:
            reader = PdfReader(self.pdf_path)
            num_pages = len(reader.pages)
            print(f"📄 Loading PDF: {os.path.basename(self.pdf_path)}")
            print(f"   Pages: {num_pages}")
            
            text_parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            print(f"✅ Extracted {len(full_text)} characters from PDF")
            
            return full_text
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise
    
    def load_with_metadata(self) -> Dict:
        """
        Load PDF with metadata
        
        Returns:
            Dictionary with text and metadata
        """
        try:
            reader = PdfReader(self.pdf_path)
            
            # Extract text
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(self.pdf_path),
                'num_pages': len(reader.pages),
                'num_characters': len(full_text),
            }
            
            # Add PDF metadata if available
            if reader.metadata:
                metadata.update({
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                })
            
            return {
                'text': full_text,
                'metadata': metadata
            }
        except Exception as e:
            print(f"❌ Error loading PDF with metadata: {e}")
            raise
    
    @staticmethod
    def load_from_path(pdf_path: str) -> str:
        """
        Static method to load PDF from path
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        loader = PDFLoader(pdf_path)
        return loader.load()
