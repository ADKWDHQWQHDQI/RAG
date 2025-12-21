"""
Main entry point for RAG system
"""
import sys
import os
from rag_pipeline import RAGPipeline
from config import Config


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 60)
    print("🚀 Local RAG System")
    print("   SQLite + Python + Open Source LLM")
    print("   ✨ Zero Installation Required!")
    print("=" * 60 + "\n")


def print_menu():
    """Print main menu"""
    print("\n" + "-" * 60)
    print("Options:")
    print("  1. Ingest PDF")
    print("  2. Ask Question")
    print("  3. Show Statistics")
    print("  4. Clear Database")
    print("  5. Exit")
    print("-" * 60)


def ingest_pdf_interactive(pipeline: RAGPipeline):
    """Interactive PDF ingestion"""
    pdf_path = input("\nEnter PDF file path: ").strip().strip('"\'')
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    pipeline.ingest_pdf(pdf_path)


def query_interactive(pipeline: RAGPipeline):
    """Interactive query"""
    question = input("\nEnter your question: ").strip()
    
    if not question:
        print("❌ Please enter a question")
        return
    
    result = pipeline.query(question)
    
    print("\n" + "=" * 60)
    print("📚 SOURCES:")
    print("=" * 60)
    for i, (content, distance) in enumerate(result['sources'], 1):
        print(f"\n[Source {i}] (distance: {distance:.4f})")
        print(content[:200] + "..." if len(content) > 200 else content)


def show_stats(pipeline: RAGPipeline):
    """Show pipeline statistics"""
    stats = pipeline.get_stats()
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"Total documents in database: {stats['total_documents']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Chunk size: {stats['chunk_size']}")
    print(f"LLM model: {stats['llm_model']}")
    print("=" * 60)


def main():
    """Main application loop"""
    print_header()
    
    # Check if setup has been run
    if not os.path.exists('.env'):
        print("⚠️  No .env file found!")
        print("   1. Copy .env.example to .env")
        print("   2. Configure your settings")
        print("   3. Run: python setup_database.py")
        return
    
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline()
        pipeline.initialize()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("\nDid you run setup_database.py?")
        return
    
    # Main loop
    try:
        while True:
            print_menu()
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                ingest_pdf_interactive(pipeline)
            elif choice == '2':
                query_interactive(pipeline)
            elif choice == '3':
                show_stats(pipeline)
            elif choice == '4':
                pipeline.clear_database()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    finally:
        pipeline.close()


def demo():
    """Quick demo function"""
    print_header()
    print("Running quick demo...\n")
    
    # Initialize pipeline
    with RAGPipeline() as pipeline:
        # Example usage
        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
            print(f"Ingesting: {pdf_path}")
            pipeline.ingest_pdf(pdf_path)
            
            # Ask a question
            question = input("\nAsk a question about the document: ")
            result = pipeline.query(question)
            
            print("\n📚 Sources used:")
            for i, (content, distance) in enumerate(result['sources'], 1):
                print(f"\n[{i}] Distance: {distance:.4f}")
                print(content[:150] + "...")
        else:
            print("Usage: python main.py <pdf_file_path>")
            print("   Or run without arguments for interactive mode")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != '--interactive':
        demo()
    else:
        main()
