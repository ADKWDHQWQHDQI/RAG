"""
LLM module for generating answers using Ollama
"""
import subprocess
import requests
import json
from typing import Optional
from config import Config


class OllamaLLM:
    """Ollama LLM wrapper"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.OLLAMA_MODEL
        self.ollama_available = self.check_ollama_installed()
        if self.ollama_available:
            self.check_ollama_running()
    
    def check_ollama_installed(self):
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            if result.returncode == 0:
                print(f"[SUCCESS] Ollama is installed: {result.stdout.strip()}")
                return True
            else:
                print("[WARNING] Ollama not found. Please install from https://ollama.ai")
                return False
        except FileNotFoundError:
            print("[WARNING] Ollama not found. Please install from https://ollama.ai")
            return False
        except Exception as e:
            print(f"[WARNING] Error checking Ollama: {e}")
            return False
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=3
            )
            if result.returncode == 0:
                print("[SUCCESS] Ollama service is running")
                return True
            else:
                print("[WARNING] Ollama service may not be running. Try 'ollama serve'")
                return False
        except subprocess.TimeoutExpired:
            print("[WARNING] Ollama service is not responding. Please start it with 'ollama serve'")
            return False
        except Exception as e:
            print(f"[WARNING] Cannot connect to Ollama service: {e}")
            return False
    
    def check_model_available(self) -> bool:
        """Check if the specified model is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )
            # Check if model name appears in the output
            lines = result.stdout.lower()
            if self.model_name.lower() in lines:
                print(f"[SUCCESS] Model available: {self.model_name}")
                return True
            else:
                print(f"[WARNING] Model '{self.model_name}' not found.")
                print(f"   Run: ollama pull {self.model_name}")
                print(f"\nAvailable models:")
                print(result.stdout)
                return False
        except Exception as e:
            print(f"[WARNING] Error checking model: {e}")
            return False
    
    def pull_model(self):
        """Pull the model from Ollama"""
        print(f"[DOWNLOADING] Pulling model: {self.model_name}")
        try:
            subprocess.run(
                ["ollama", "pull", self.model_name],
                check=True
            )
            print(f"[SUCCESS] Model pulled successfully: {self.model_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error pulling model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Generate response from LLM using Ollama HTTP API (much faster!)
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if error
        """
        if not self.ollama_available:
            print("[ERROR] Ollama is not available")
            return None
        
        try:
            # Use Ollama HTTP API - keeps model loaded, much faster!
            ollama_url = getattr(Config, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 300,   # Sufficient tokens for complete answers
                        "temperature": 0.0,   # Zero temp for deterministic, factual responses
                        "top_p": 0.8,
                        "top_k": 10,
                        "num_ctx": 4096,      # Larger context window
                        "repeat_penalty": 1.1
                    }
                },
                timeout=90  # 90 second timeout for llm with longer context
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            elif response.status_code == 404:
                print(f"[ERROR] Model '{self.model_name}' not found in Ollama")
                print(f"[INFO] Run: ollama pull {self.model_name}")
                print(f"[INFO] Or check available models: ollama list")
                return None
            else:
                print(f"[ERROR] Error from Ollama API: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print("[ERROR] LLM generation timed out (45s limit)")
            print("[INFO] Model may be slow. Try restarting: ollama serve")
            return None
        except requests.exceptions.ConnectionError:
            print("[ERROR] Cannot connect to Ollama service")
            print("[INFO] Make sure Ollama is running. Try 'ollama serve' in another terminal")
            return None
        except Exception as e:
            print(f"[ERROR] Error calling Ollama: {e}")
            return None
    
    def ask_with_context(self, context: str, question: str) -> Optional[str]:
        """
        Ask a question with context (RAG pattern)
        
        Args:
            context: Retrieved context from vector search
            question: User's question
            
        Returns:
            Generated answer
        """
        # Keep more context for accurate answers
        max_context_length = 4000  # Increased to ensure all country mentions are included
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Read the information below carefully and answer the question based ONLY on what you find in this text.

Information:
{context}

Question: {question}

Answer (extract directly from the information above):"""
        
        return self.generate(prompt, max_tokens=100)
    
    def stream_generate(self, prompt: str):
        """
        Stream generation (for future enhancement)
        Note: This is a simple implementation. For real streaming, 
        you might want to use the Ollama Python library instead.
        """
        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=prompt)
            
            if process.returncode == 0:
                return stdout.strip()
            else:
                print(f"[ERROR] Error: {stderr}")
                return None
        except Exception as e:
            print(f"[ERROR] Error streaming: {e}")
            return None
