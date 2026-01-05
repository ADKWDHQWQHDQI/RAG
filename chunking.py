"""
Text chunking module for splitting documents
"""
from typing import List, Optional
from config import Config


class TextChunker:
    """Text chunking with overlap"""
    
    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.overlap = overlap or Config.CHUNK_OVERLAP
        
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            start += self.chunk_size - self.overlap
        
        print(f"[SUCCESS] Created {len(chunks)} chunks (size={self.chunk_size}, overlap={self.overlap})")
        return chunks
    
    def chunk_text_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences for better semantic boundaries
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        print(f"[SUCCESS] Created {len(chunks)} sentence-based chunks")
        return chunks
    
    @staticmethod
    def chunk_by_tokens(text: str, max_tokens: int = 600, overlap_tokens: int = 50) -> List[str]:
        """
        Chunk text by approximate token count (words)
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap in tokens
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + max_tokens
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            start += max_tokens - overlap_tokens
        
        print(f"[SUCCESS] Created {len(chunks)} token-based chunks")
        return chunks
