"""
Adaptive RAG using LangGraph patterns with SQLite and Ollama
"""
from typing import List, Dict, Literal, TypedDict, Optional, Tuple
from database import Database
from embeddings import EmbeddingModel
from llm import OllamaLLM
from config import Config
import json
import logging
from datetime import datetime


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add web search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


class AdaptiveRAGAgent:
    """
    Adaptive RAG Agent that combines:
    - Query routing (vectorstore vs web search)
    - Document grading (relevance checking)
    - Web search
    - Query transformation
    - Answer grading (hallucination check - DYNAMIC based on datasource)
    """
    
    def __init__(self, db: Database, embedder: EmbeddingModel, llm: OllamaLLM, verbose: bool = False):
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.max_retries = Config.MAX_RETRIES
        self.verbose = verbose
        
        # Setup logging for query tracking
        self.query_log = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for tracking queries and results."""
        self.logger = logging.getLogger('AdaptiveRAG')
        
        # Only add handler if not already configured
        if not self.logger.handlers:
            handler = logging.FileHandler('rag_queries.log')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log_query(self, question: str, route: str, num_sources: int, success: bool, retries: int = 0):
        """Log query information for analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'route_taken': route,
            'num_sources': num_sources,
            'success': success,
            'retries_used': retries
        }
        self.query_log.append(log_entry)
        
        # Prevent memory leak - keep only last 1000 queries
        if len(self.query_log) > 1000:
            self.query_log = self.query_log[-1000:]
        
        self.logger.info(f"Query: {question[:50]}... | Route: {route} | Sources: {num_sources} | Success: {success}")
    
    def route_question(self, question: str) -> Literal["vectorstore", "websearch"]:
        """Route question to vectorstore or web search."""
        if self.verbose:
            print("---ROUTE QUESTION---")
        
        # Check if we have documents in the database
        try:
            doc_count = self.db.get_document_count()
            has_documents = doc_count > 0
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to check document count: {e}")
            has_documents = False
        
        # If no documents, always use web search
        if not has_documents:
            if self.verbose:
                print("---NO DOCUMENTS IN DB, USING WEB SEARCH---")
            else:
                print("[ROUTING] No local documents, using web search...")
            return 'websearch'
        
        # ALWAYS try vectorstore first - let document grading decide if relevant
        if self.verbose:
            print("---ROUTE QUESTION TO VECTORSTORE (will fallback to web if irrelevant)---")
        else:
            print("[ROUTING] Checking local documents first...")
        return 'vectorstore'
    
    def retrieve(self, question: str) -> List[Tuple[str, float]]:
        """Retrieve documents from vectorstore with their similarity distances."""
        if self.verbose:
            print("---RETRIEVE FROM VECTORSTORE---")
        else:
            print("[SEARCH] Searching local documents...")
        
        try:
            query_embedding_result = self.embedder.encode(question)
            
            if not query_embedding_result:
                if self.verbose:
                    print("[ERROR] Failed to embed question")
                else:
                    print("[ERROR] Failed to process query")
                return []
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Embedding error: {e}")
            else:
                print("[ERROR] Failed to process query")
            return []
        
        try:
            # Ensure single embedding - safely check if it's a batch result
            if isinstance(query_embedding_result, list) and len(query_embedding_result) > 0 and isinstance(query_embedding_result[0], list):
                # It's a batch result [[0.1, 0.2, ...]], take the first embedding
                query_embedding: List[float] = query_embedding_result[0]
            else:
                # It's already a single embedding [0.1, 0.2, ...]
                from typing import cast
                query_embedding: List[float] = cast(List[float], query_embedding_result)
            
            results = self.db.similarity_search(query_embedding, k=Config.TOP_K)
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Database search error: {e}")
            else:
                print("[ERROR] Search failed")
            return []
        
        if not results:
            if self.verbose:
                print("[ERROR] No relevant documents found")
            else:
                print("[INFO] No relevant documents found")
            return []
        
        if self.verbose:
            print(f"[SUCCESS] Retrieved {len(results)} documents")
        else:
            print(f"[SUCCESS] Found {len(results)} relevant documents")
        
        return results
    
    def grade_documents(self, question: str, documents: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], str]:
        """Grade documents for relevance and determine if web search is needed."""
        if self.verbose:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        else:
            print("[GRADING] Grading document relevance...")
        
        if not documents:
            return [], "Yes"
        
        filtered_docs = []
        
        # Extract just the content for grading
        doc_contents = [content for content, _ in documents]
        
        # Batch grading: grade all documents in one call for efficiency
        try:
            docs_preview = "\n\n".join([f"Document {i+1}: {doc[:400]}..." for i, doc in enumerate(doc_contents)])
            
            # Simplified grading prompt for better LLM compliance
            batch_grader_prompt = f"""You are a document relevance checker. For each document, answer "yes" if it's relevant to the question, "no" if not.

Question: {question}

Documents:
{docs_preview}

Respond with ONLY a JSON array like ["yes","no","yes","yes","no"] with exactly {len(documents)} values.

JSON array:"""
            
            # Increased token limit to ensure complete response
            response = self.llm.generate(batch_grader_prompt, max_tokens=300)
            
            if response:
                json_str = ""  # Initialize outside try block
                try:
                    # Clean response to extract JSON array
                    response_clean = response.strip()
                    
                    # Debug: Show actual response for troubleshooting
                    if not self.verbose:
                        print(f"[DEBUG] Grader raw response: {response_clean[:200]}")
                    
                    # Find JSON array in response
                    start = response_clean.find('[')
                    end = response_clean.rfind(']')
                    
                    if start != -1 and end != -1:
                        json_str = response_clean[start:end+1]
                        scores = json.loads(json_str)
                        
                        if isinstance(scores, list) and len(scores) == len(documents):
                            relevant_count = 0
                            for (content, distance), score in zip(documents, scores):
                                score_str = str(score).lower().strip().replace('"', '').replace("'", '')
                                if score_str == 'yes':
                                    filtered_docs.append((content, distance))
                                    relevant_count += 1
                            
                            if self.verbose:
                                print(f"[INFO] Grader marked {relevant_count}/{len(documents)} documents as relevant")
                            elif relevant_count == 0:
                                # Show debug info when all rejected
                                print(f"[DEBUG] Grader response: {response_clean[:150]}")
                                print(f"[DEBUG] All documents rejected as irrelevant")
                        else:
                            print(f"[WARNING] Score count mismatch: got {len(scores) if isinstance(scores, list) else 'invalid'}, expected {len(documents)}")
                            print(f"[DEBUG] Response was: {response_clean[:200]}")
                            # Fallback: Use keyword matching
                            filtered_docs = self._fallback_keyword_grading(question, doc_contents, documents)
                    else:
                        print("[WARNING] No JSON array found in grader response")
                        print(f"[DEBUG] Response was: {response_clean[:200]}")
                        # Fallback: Use keyword matching
                        filtered_docs = self._fallback_keyword_grading(question, doc_contents, documents)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON parse error: {e}")
                    print(f"[DEBUG] Attempted to parse: {json_str[:150] if json_str else 'N/A'}")
                    # Fallback: Use keyword matching
                    filtered_docs = self._fallback_keyword_grading(question, doc_contents, documents)
            else:
                if self.verbose:
                    print("[WARNING] Empty grader response")
                # Fallback: reject all to trigger web search
                print("[WARNING] No grading response, will try web search")
                filtered_docs = []
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Grading error: {e}")
            # On error, reject all to trigger web search
            print(f"[ERROR] Grading failed: {str(e)[:50]}, will try web search")
            filtered_docs = []
        
        web_search = "Yes" if not filtered_docs else "No"
        
        if not self.verbose:
            if filtered_docs:
                print(f"[SUCCESS] {len(filtered_docs)} documents are relevant")
            else:
                print("[INFO] No relevant documents found, will try web search...")
        
        return filtered_docs, web_search
    
    def _fallback_keyword_grading(self, question: str, doc_contents: List[str], doc_tuples: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Fallback grading using simple keyword matching when LLM grading fails."""
        print("[INFO] Using keyword-based fallback grading...")
        
        # Extract key terms from question (remove common stop words)
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'are', 'do', 'does', 'can', 'of', 'in', 'to', 'for', 'on', 'with', 'about'}
        question_words = set(question.lower().split()) - stop_words
        
        filtered_docs = []
        for (content, distance), doc_content in zip(doc_tuples, doc_contents):
            doc_lower = doc_content.lower()
            # Check if any question keywords appear in document
            matches = sum(1 for word in question_words if word in doc_lower)
            # Accept if at least 30% of keywords match
            if matches >= len(question_words) * 0.3 or matches >= 2:
                filtered_docs.append((content, distance))
        
        print(f"[INFO] Keyword matching found {len(filtered_docs)}/{len(doc_tuples)} relevant documents")
        return filtered_docs
    
    def web_search(self, question: str) -> List[Tuple[str, float]]:
        """Perform web search using DuckDuckGo."""
        if self.verbose:
            print("---WEB SEARCH---")
        else:
            print("[WEB SEARCH] Searching the web...")
        
        if not Config.ENABLE_WEB_SEARCH:
            if self.verbose:
                print("[ERROR] Web search is disabled in config")
            else:
                print("[INFO] Web search is disabled in configuration")
            return []
        
        try:
            from duckduckgo_search import DDGS
            
            if self.verbose:
                print("[SEARCH] Searching with DuckDuckGo...")
            
            ddgs = DDGS()
            # Increased from 3 to 5 results for better coverage
            results = ddgs.text(
                question,
                backend='lite',
                max_results=5
            )
            
            results_list = list(results) if results else []
            
            if results_list:
                web_docs = []
                for result in results_list:
                    title = result.get('title', '')
                    body = result.get('body', '')
                    href = result.get('href', '')
                    
                    # Better formatting with clear source attribution
                    content = f"**{title}**\nSource: {href}\n\n{body}"
                    # Web results have no meaningful distance, use 1.0 (max distance)
                    web_docs.append((content, 1.0))
                
                if self.verbose:
                    print(f"[SUCCESS] Found {len(web_docs)} web results")
                else:
                    print(f"[SUCCESS] Found {len(web_docs)} web results")
                
                return web_docs
            else:
                if self.verbose:
                    print("[ERROR] No web results found")
                else:
                    print("[INFO] No web results found for this query")
                return []
                
        except ImportError:
            if self.verbose:
                print("[ERROR] duckduckgo-search not installed")
            else:
                print("[ERROR] Web search library not available. Install: pip install duckduckgo-search")
            return []
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Web search failed: {e}")
            else:
                print(f"[ERROR] Web search error: {str(e)[:50]}")
            return []
    

    def transform_query(self, question: str) -> str:
        """Transform query to produce a better question."""
        if self.verbose:
            print("---TRANSFORM QUERY---")
        else:
            print("[TRANSFORM] Refining query...")
        
        transform_prompt = f"""You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.

Look at the initial question and formulate an improved question.

Here is the initial question: {question}

Improved question with no preamble:"""
        
        response = self.llm.generate(transform_prompt, max_tokens=100)
        
        if response:
            if self.verbose:
                print(f"Original: {question}")
                print(f"Improved: {response}")
            return response
        
        return question
    
    def generate_answer(self, question: str, documents: List[Tuple[str, float]], include_sources: bool = False) -> str:
        """Generate answer using retrieved documents with optional source citations."""
        if self.verbose:
            print("---GENERATE ANSWER---")
        else:
            print("[GENERATE] Generating answer...")
        
        if not documents:
            return "I don't have enough information to answer that question."
        
        try:
            # Extract content from tuples
            doc_contents = [content for content, _ in documents]
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_contents)])
            
            # Increase context window to 5000 tokens (from 3000)
            max_context_tokens = 5000
            max_context_length = max_context_tokens * 4
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            gen_prompt = f"""You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Use three sentences maximum and keep the answer concise.

Question: {question}

Context:
{context}

Answer:"""
            
            # Increased token limit from 200 to 300 for complete answers
            answer = self.llm.generate(gen_prompt, max_tokens=300)
            
            if answer:
                # Add source attribution if requested
                if include_sources and len(documents) > 0:
                    answer += f"\n\n[Based on {len(documents)} source(s)]"
                return answer
            
            return "Sorry, I couldn't generate an answer."
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Answer generation failed: {e}")
            return "Sorry, an error occurred while generating the answer."
    
    def grade_generation_v_documents(self, question: str, documents: List[Tuple[str, float]], generation: str) -> Literal["useful", "not useful", "not supported"]:
        """Grade the generation for hallucinations and usefulness."""
        if self.verbose:
            print("---CHECK HALLUCINATIONS---")
        
        # Extract content from tuples
        doc_contents = [content for content, _ in documents]
        context = "\n\n".join(doc_contents)
        
        hallucination_prompt = f"""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Set of facts:
{context[:2000]}

LLM generation: {generation}

Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

Return a JSON object with a single key 'score' that is either 'yes' or 'no'.

JSON Response:"""
        
        response = self.llm.generate(hallucination_prompt, max_tokens=10)
        
        if response:
            try:
                result = json.loads(response)
                score = result.get('score', 'no').lower()
            except json.JSONDecodeError:
                score = 'yes' if 'yes' in response.lower() else 'no'
            
            if score == 'yes':
                if self.verbose:
                    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            else:
                if self.verbose:
                    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
                return "not supported"
        
        # Check if generation addresses question
        if self.verbose:
            print("---GRADE GENERATION vs QUESTION---")
        
        relevance_prompt = f"""You are a grader assessing whether an answer addresses / resolves a question.

User question: {question}

LLM generation: {generation}

Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.

Return a JSON object with a single key 'score' that is either 'yes' or 'no'.

JSON Response:"""
        
        response = self.llm.generate(relevance_prompt, max_tokens=10)
        
        if response:
            try:
                result = json.loads(response)
                score = result.get('score', 'no').lower()
            except json.JSONDecodeError:
                score = 'yes' if 'yes' in response.lower() else 'no'
            
            if score == 'yes':
                if self.verbose:
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                if self.verbose:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        
        return "useful"
    
    def run(self, question: str) -> Dict:
        """Run the adaptive RAG workflow with DYNAMIC answer grading and retry logic."""
        # Input validation
        if not question or not isinstance(question, str):
            return {
                'answer': "Invalid question. Please provide a valid text question.",
                'sources': [],
                'num_sources': 0,
                'question': '',
                'route_taken': 'error'
            }
        
        # Sanitize input
        original_question = question.strip()
        question = original_question
        
        # Check max length (prevent abuse)
        max_question_length = 1000
        if len(question) > max_question_length:
            question = question[:max_question_length]
            if self.verbose:
                print(f"[WARNING] Question truncated to {max_question_length} characters")
        
        if not self.verbose:
            print(f"\n[PROCESSING] {question}")
            print("-" * 60)
        else:
            print("\n" + "=" * 60)
            print(f"QUESTION: {question}")
            print("=" * 60)
        
        # Retry loop with query transformation
        retries_remaining = self.max_retries
        documents = []
        route_taken = "unknown"
        datasource = "vectorstore"  # Initialize to prevent unbound error
        enable_grading = False  # Initialize to prevent unbound error
        
        while retries_remaining > 0:
            datasource = self.route_question(question)
            route_taken = datasource
            enable_grading = (datasource == "vectorstore") and Config.ENABLE_ANSWER_GRADING
            
            if self.verbose:
                if enable_grading:
                    print("---ANSWER GRADING: ENABLED (local documents)---")
                else:
                    print("---ANSWER GRADING: DISABLED (web search snippets)---")
            
            if datasource == "websearch":
                documents = self.web_search(question)
                
                if not documents:
                    if self.verbose:
                        print("---WEB SEARCH FAILED, TRYING VECTORSTORE---")
                    else:
                        print("[INFO] Web search failed, trying local documents...")
                    
                    documents = self.retrieve(question)
                    if documents:
                        filtered_docs, _ = self.grade_documents(question, documents)
                        documents = filtered_docs
                        enable_grading = Config.ENABLE_ANSWER_GRADING
                        route_taken = "vectorstore"
            else:
                documents = self.retrieve(question)
                
                if documents:
                    filtered_docs, need_web_search = self.grade_documents(question, documents)
                    documents = filtered_docs
                    
                    if need_web_search == "Yes" and not documents:
                        if self.verbose:
                            print("---DECISION: ALL DOCUMENTS NOT RELEVANT, FALLBACK TO WEB SEARCH---")
                        else:
                            print("\n[INFO] ⚠️  Question not found in local documents")
                            print("[INFO] 🔍 Proceeding to web search...\n")
                        
                        web_docs = self.web_search(question)
                        documents.extend(web_docs)
                        enable_grading = False
                        route_taken = "websearch"
            
            # If we have documents, break the retry loop
            if documents:
                break
            
            # No documents found, try query transformation if retries remain
            retries_remaining -= 1
            if retries_remaining > 0:
                if self.verbose:
                    print(f"---RETRY {self.max_retries - retries_remaining}/{self.max_retries}: TRANSFORMING QUERY---")
                else:
                    print(f"[INFO] Retrying with refined query... ({self.max_retries - retries_remaining}/{self.max_retries})")
                
                question = self.transform_query(question)
            else:
                if self.verbose:
                    print("---MAX RETRIES REACHED---")
                break
        
        if not documents:
            if self.verbose:
                print("---NO DOCUMENTS AVAILABLE---")
            else:
                print("[INFO] No information found after all retries")
            
            if datasource == "websearch":
                answer = ("I couldn't retrieve information from web search. "
                         "Please check your internet connection or try a different question.")
            else:
                answer = ("I don't have any documents in the knowledge base yet. "
                         "Please ingest some PDFs first or enable web search for real-time information.")
            
            if not self.verbose:
                print(f"\n[ANSWER] {answer}\n")
            
            # Log failed query
            self._log_query(original_question, route_taken, 0, False, self.max_retries)
            
            return {
                'answer': answer,
                'sources': [],
                'num_sources': 0,
                'question': original_question,
                'route_taken': route_taken
            }
        
        # Generate answer with source citations
        generation = self.generate_answer(question, documents, include_sources=True)
        
        # Apply hallucination checking if enabled and using local documents
        if enable_grading and Config.ENABLE_ANSWER_GRADING:
            if self.verbose:
                print("---CHECKING ANSWER QUALITY---")
            
            grade = self.grade_generation_v_documents(question, documents, generation)
            
            if grade == "not supported":
                if self.verbose:
                    print("---WARNING: ANSWER NOT GROUNDED IN DOCUMENTS---")
                else:
                    print("[WARNING] Answer may contain unsupported information")
                generation = "I found relevant documents but cannot provide a confident answer based solely on them. " + generation
            elif grade == "not useful":
                if self.verbose:
                    print("---WARNING: ANSWER DOESN'T ADDRESS QUESTION---")
                else:
                    print("[WARNING] Answer may not fully address the question")
                # Retry once with transformed query if not useful
                if retries_remaining > 0:
                    question = self.transform_query(original_question)
                    documents = self.retrieve(question)
                    if documents:
                        filtered_docs, _ = self.grade_documents(question, documents)
                        if filtered_docs:
                            generation = self.generate_answer(question, filtered_docs, include_sources=True)
        
        if not self.verbose:
            print("\n" + "-" * 60)
            print("[ANSWER]")
            print(generation)
            print("-" * 60)
            print(f"[ROUTE] Via: {route_taken}")
            print("-" * 60 + "\n")
        else:
            print("\n" + "=" * 60)
            print("FINAL ANSWER:")
            print(generation)
            print("=" * 60 + "\n")
        
        # Log the query for tracking and analysis
        retries_used = self.max_retries - retries_remaining
        self._log_query(original_question, route_taken, len(documents), True, retries_used)
        
        return {
            'answer': generation,
            'sources': documents,
            'num_sources': len(documents),
            'question': original_question,
            'route_taken': route_taken
        }
    
    def get_query_stats(self) -> Dict:
        """Get statistics about queries processed."""
        if not self.query_log:
            return {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_sources': 0,
                'route_breakdown': {},
                'avg_retries': 0
            }
        
        total = len(self.query_log)
        successful = sum(1 for q in self.query_log if q['success'])
        failed = total - successful
        
        avg_sources = sum(q['num_sources'] for q in self.query_log) / total if total > 0 else 0
        avg_retries = sum(q['retries_used'] for q in self.query_log) / total if total > 0 else 0
        
        # Route breakdown
        route_counts = {}
        for q in self.query_log:
            route = q['route_taken']
            route_counts[route] = route_counts.get(route, 0) + 1
        
        return {
            'total_queries': total,
            'successful_queries': successful,
            'failed_queries': failed,
            'success_rate': f"{(successful/total*100):.1f}%" if total > 0 else "0%",
            'avg_sources': round(avg_sources, 2),
            'avg_retries': round(avg_retries, 2),
            'route_breakdown': route_counts
        }
    
    def print_stats(self):
        """Print query statistics in a readable format."""
        stats = self.get_query_stats()
        print("\n" + "=" * 60)
        print("[QUERY STATISTICS]")
        print("=" * 60)
        print(f"Total Queries:       {stats['total_queries']}")
        print(f"Successful:          {stats['successful_queries']} ({stats['success_rate']})")
        print(f"Failed:              {stats['failed_queries']}")
        print(f"Avg Sources/Query:   {stats['avg_sources']}")
        print(f"Avg Retries/Query:   {stats['avg_retries']}")
        print(f"\nRoute Breakdown:")
        for route, count in stats['route_breakdown'].items():
            print(f"  - {route}: {count}")
        print("=" * 60 + "\n")
