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


class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(self):
        self.db = Database()
        self.embedder = EmbeddingModel()
        self.chunker = TextChunker()
        self.llm = OllamaLLM()
        self.initialized = False
    
    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing RAG Pipeline...")
        
        # Connect to database
        if not self.db.connect():
            raise RuntimeError("Failed to connect to database")
        
        # Create extension and table
        self.db.create_extension()
        self.db.create_table()
        
        # Load embedding model
        if not self.embedder.load():
            raise RuntimeError("Failed to load embedding model")
        
        # Check LLM
        self.llm.check_model_available()
        
        self.initialized = True
        print("✅ RAG Pipeline initialized\n")
    
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
            
            # Step 2: Chunk text
            chunks = self.chunker.chunk_text(text)
            print(f"📝 Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("🔄 Generating embeddings...")
            embeddings = self.embedder.encode_batch(chunks, batch_size=32)
            
            if not embeddings:
                raise RuntimeError("Failed to generate embeddings")
            
            # Step 4: Store in database
            print("💾 Storing in database...")
            documents = [
                (chunk, embedding, metadata)
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            doc_ids = self.db.insert_documents_batch(documents)
            
            if doc_ids:
                print(f"✅ Successfully ingested {len(doc_ids)} chunks")
                print(f"📊 Total documents in DB: {self.db.get_document_count()}")
                return True
            else:
                print("❌ Failed to store documents")
                return False
                
        except Exception as e:
            print(f"❌ Error ingesting PDF: {e}")
            return False
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        top_k = top_k or Config.TOP_K
        
        print(f"\n❓ QUERY: {question}")
        print("=" * 60)
        
        try:
            # Step 1: Embed the question
            print("🔄 Embedding question...")
            query_embedding_result = self.embedder.encode(question)
            
            if not query_embedding_result:
                raise RuntimeError("Failed to embed question")
            
            # Ensure it's a single embedding (List[float]), not a batch
            if isinstance(query_embedding_result[0], list):
                raise RuntimeError("Expected single embedding, got batch")
            
            # Type narrowing: we know it's List[float] at this point
            query_embedding: List[float] = query_embedding_result  # type: ignore
            
            # Step 2: Similarity search
            print(f"🔍 Searching for top {top_k} similar documents...")
            results = self.db.similarity_search(query_embedding, k=top_k)
            
            if not results:
                print("⚠️  No relevant documents found")
                return {
                    'answer': "I don't have any relevant information to answer that question.",
                    'sources': [],
                    'num_sources': 0
                }
            
            print(f"✅ Found {len(results)} relevant documents")
            
            # Step 3: Prepare context
            context_parts = []
            for i, (content, distance) in enumerate(results, 1):
                context_parts.append(f"[Document {i}]\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Step 4: Generate answer using LLM
            print("🤖 Generating answer...")
            answer = self.llm.ask_with_context(context, question)
            
            if not answer:
                answer = "Sorry, I couldn't generate an answer. Please check if Ollama is running and the model is available."
            
            print("\n" + "=" * 60)
            print("💡 ANSWER:")
            print(answer)
            print("=" * 60)
            
            return {
                'answer': answer,
                'sources': results,
                'num_sources': len(results),
                'question': question
            }
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
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
        confirm = input("⚠️  This will delete all documents. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            self.db.clear_all_documents()
        else:
            print("Cancelled")
    
    def close(self):
        """Close all connections"""
        if self.db:
            self.db.close()
        print("✅ Pipeline closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
