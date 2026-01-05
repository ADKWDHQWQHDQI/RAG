"""
Main RAG pipeline orchestration
"""
from typing import List, Dict, Optional
from database import Database
from embeddings import EmbeddingModel
from pdf_loader import PDFLoader
from chunking import TextChunker
from llm import OllamaLLM
from config import Config
from adaptive_rag_graph import AdaptiveRAGAgent



class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(self):
        self.db = Database()
        self.embedder = EmbeddingModel()
        self.chunker = TextChunker()
        self.llm = OllamaLLM()
        self.rag_agent = None
        self.initialized = False
    
    def initialize(self):
        """Initialize all components"""
        print("[INITIALIZING] Initializing RAG Pipeline...")
        
        # Connect to database
        if not self.db.connect():
            raise RuntimeError("Failed to connect to database")
        
        # Create extension and table
        self.db.create_extension()
        self.db.create_table()
        
        # Load embedding model
        if not self.embedder.load():
            raise RuntimeError("Failed to load embedding model")
        
        # Validate embedding dimensions match config
        actual_dim = self.embedder.get_embedding_dimension()
        expected_dim = Config.EMBEDDING_DIM
        if actual_dim != expected_dim:
            print(f"[WARNING] Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            print(f"[INFO] Using actual dimension: {actual_dim}")
        
        # Check LLM availability
        if not self.llm.ollama_available:
            raise RuntimeError("Ollama is not available. Please install and start Ollama.")
        
        self.llm.check_model_available()
        
        # Initialize Adaptive RAG Agent with hallucination checking
        self.rag_agent = AdaptiveRAGAgent(
            embedder=self.embedder,
            db=self.db,
            llm=self.llm,
            verbose=False
        )
        
        self.initialized = True
        print("[SUCCESS] RAG Pipeline initialized\n")
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """
        Ingest a PDF document into the vector database
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Success status
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        print(f"\n📥 INGESTING PDF: {pdf_path}")
        print("=" * 60)
        
        try:
            # Step 1: Load PDF
            loader = PDFLoader(pdf_path)
            pdf_data = loader.load_with_metadata()
            text = pdf_data['text']
            metadata = pdf_data['metadata']
            
            # Step 2: Chunk text using sentence-based chunking for better semantic boundaries
            try:
                chunks = self.chunker.chunk_text_sentences(text)
                print(f"[INFO] Created {len(chunks)} sentence-based chunks")
            except Exception as e:
                # Fallback to simple chunking if sentence chunking fails
                print(f"[WARNING] Sentence chunking failed: {e}, using simple chunking")
                chunks = self.chunker.chunk_text(text)
                print(f"[INFO] Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("[PROCESSING] Generating embeddings...")
            embeddings = self.embedder.encode_batch(chunks, batch_size=32)
            
            if not embeddings:
                raise RuntimeError("Failed to generate embeddings")
            
            # Step 4: Store in database
            print("[STORING] Storing in database...")
            documents = [
                (chunk, embedding, metadata)
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            doc_ids = self.db.insert_documents_batch(documents)
            
            if doc_ids:
                print(f"[SUCCESS] Successfully ingested {len(doc_ids)} chunks")
                print(f"[INFO] Total documents in DB: {self.db.get_document_count()}")
                return True
            else:
                print("[ERROR] Failed to store documents")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error ingesting PDF: {e}")
            return False
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Query the RAG system using Adaptive RAG with hallucination checking
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        print(f"\n❓ QUERY: {question}")
        print("=" * 60)
        
        try:
            if self.rag_agent is None:
                raise RuntimeError("RAG agent not initialized")
            # Use Adaptive RAG Agent for intelligent routing and validation
            result = self.rag_agent.run(question)
            
            # Sources already include distances from the agent
            return {
                'answer': result['answer'],
                'sources': result['sources'],  # Already list of (content, distance) tuples
                'num_sources': result['num_sources'],
                'question': result['question']
            }
            
        except Exception as e:
            print(f"[ERROR] Error during query: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'num_sources': 0
            }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'total_documents': self.db.get_document_count(),
            'embedding_model': self.embedder.model_name,
            'embedding_dim': Config.EMBEDDING_DIM,
            'chunk_size': Config.CHUNK_SIZE,
            'llm_model': self.llm.model_name
        }
    
    def clear_database(self):
        """Clear all documents from database"""
        confirm = input("[WARNING] This will delete all documents. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            self.db.clear_all_documents()
        else:
            print("Cancelled")
    
    def close(self):
        """Close all connections"""
        if self.db:
            self.db.close()
        print("[INFO] Pipeline closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
