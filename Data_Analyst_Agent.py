'''
If for privacy reason you don't trust the app use your own API key.
Change the API key at ".env"
If you can`t contact the team at support@archusers.com (we'll send you the key)
'''

import os
import io
import json
import base64
import sys
import warnings
import time
from typing import Dict, List, Any, Optional, Union
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not found. Please install: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

# Core data processing imports with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Core data libraries loaded successfully")
except ImportError as e:
    print(f"❌ Error importing data libraries: {e}")
    print("\n🔧 To fix this issue, please run one of the following:")
    print("   pip uninstall numpy pandas -y")
    print("   pip install numpy==1.24.3")
    print("   pip install pandas==1.5.3")
    print("   pip install -r requirements.txt")
    input("Press Enter to exit...")
    sys.exit(1)

# File processing imports
try:
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract
    import requests
    import openai
    import re
    from datetime import datetime, timedelta
    import hashlib
    from dataclasses import dataclass
    from enum import Enum
    import difflib
except ImportError as e:
    print(f"Error importing file processing libraries: {e}")
    print("Please run: pip install PyPDF2 python-docx Pillow pytesseract requests openai")
    sys.exit(1)

# NLP imports with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        # Try to load the English model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not installed, try to install it at runtime
        print("🔧 Attempting to download spaCy English model...")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                          check=True, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy English model downloaded successfully")
        except Exception as install_error:
            nlp = None
            print(f"⚠️  Could not install spaCy model: {install_error}")
            print("Some NLP features will be limited.")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("⚠️  spaCy not available. NLP features will be limited.")

def ensure_spacy_model():
    """Ensure spaCy model is available, with runtime installation fallback"""
    global nlp, SPACY_AVAILABLE, spacy
    
    if not SPACY_AVAILABLE:
        return False
        
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            # Try installing at runtime (for Streamlit Cloud)
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                              check=True, capture_output=True)
                nlp = spacy.load("en_core_web_sm")
                return True
            except:
                return False
    return True

# UI imports
try:
    import streamlit as st
    import subprocess
except ImportError as e:
    print(f"Error importing UI libraries: {e}")
    print("Please run: pip install streamlit")
    sys.exit(1)

def test_api_key(api_key: str) -> tuple[bool, str]:
    """Test an API key without full agent initialization"""
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.getenv('OPENROUTER_APP_URL', 'https://github.com/DevanshSrajput/Dataa_Analyst_Agent'),
                "X-Title": os.getenv('OPENROUTER_APP_NAME', 'AI Document Analyst v2.0')
            }
        )
        
        response = client.chat.completions.create(
            model='meta-llama/llama-3.1-8b-instruct:free',
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
            temperature=0.1
        )
        return True, "✅ API key is valid!"
    except Exception as e:
        if "401" in str(e):
            return False, "❌ Invalid or expired API key"
        elif "429" in str(e):
            return False, "⏰ Rate limit reached, but key appears valid"
        else:
            return False, f"❌ Connection error: {str(e)}"

class DocumentAnalystAgent:
    """
    An intelligent document analysis agent that can process multiple file formats,
    perform data analysis, generate visualizations, and answer questions.
    """
    
    def __init__(self, api_key: str):
        try:
            if not api_key or api_key.strip() == "":
                raise ValueError("API key cannot be empty")
            
            # Initialize OpenRouter client
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": os.getenv('OPENROUTER_APP_URL', 'https://github.com/DevanshSrajput/Dataa_Analyst_Agent'),
                    "X-Title": os.getenv('OPENROUTER_APP_NAME', 'AI Document Analyst v2.0')
                }
            )
            
            # Set default model - using a reliable free model
            self.model = os.getenv('DEFAULT_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
            self.backup_model = os.getenv('BACKUP_MODEL', 'openai/gpt-4o-mini')
            
            self.document_content = {}
            self.data_frames = {}
            self.analysis_results = {}
            self.conversation_history = []
            
            # Initialize legal analyzer
            self.legal_analyzer = LegalDocumentAnalyzer(api_key)
            self.legal_analysis_results = {}
            self.security_manager = LegalSecurityManager()
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Test API connection
            self._test_api_connection()
            
        except Exception as e:
            raise Exception(f"Failed to initialize DocumentAnalystAgent: {str(e)}")
    
    def _test_api_connection(self):
        """Test API connection with a minimal request"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0.1
            )
            return True
        except Exception as e:
            # Try backup model if primary fails
            try:
                response = self.client.chat.completions.create(
                    model=self.backup_model,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5,
                    temperature=0.1
                )
                self.model = self.backup_model  # Switch to backup model
                print(f"Primary model failed, switched to backup model: {self.backup_model}")
                return True
            except Exception as backup_error:
                raise Exception(f"API connection test failed for both models. Primary: {str(e)}, Backup: {str(backup_error)}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def load_structured_data(self, file_path: str) -> pd.DataFrame:
        """Load structured data from CSV or Excel files"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported structured data format")
        except Exception as e:
            print(f"Error loading structured data: {str(e)}")
            return pd.DataFrame()
    
    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Process a document based on its type and extract relevant information
        
        Args:
            file_path: Path to the file
            file_name: Name of the file
            
        Returns:
            Dictionary containing processed information
        """
        file_extension = file_name.lower().split('.')[-1]
        result = {
            'file_name': file_name,
            'file_type': file_extension,
            'content': '',
            'data_frame': None,
            'summary': ''
        }
        
        try:
            if file_extension == 'pdf':
                result['content'] = self.extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                result['content'] = self.extract_text_from_docx(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['content'] = f.read()
            elif file_extension in ['csv', 'xlsx', 'xls']:
                df = self.load_structured_data(file_path)
                result['data_frame'] = df
                result['content'] = df.to_string()
                self.data_frames[file_name] = df
            elif file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                result['content'] = self.extract_text_from_image(file_path)
            
            # Store processed content
            self.document_content[file_name] = result
            
            # Generate initial summary
            result['summary'] = self.generate_document_summary(result)
            
            # Perform legal analysis if document contains legal content
            if self.is_legal_document(result['content']):
                legal_analysis = self.legal_analyzer.analyze_legal_document(
                    result['content'], 
                    self.detect_legal_document_type(result['content'])
                )
                result['legal_analysis'] = legal_analysis
                self.legal_analysis_results[file_name] = legal_analysis
            
            return result
            
        except Exception as e:
            result['content'] = f"Error processing file: {str(e)}"
            return result
    
    def _extract_response_content(self, response) -> str:
        """Extract content from Together API response"""
        try:
            # Handle Together API response format
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    # Handle case where content is a list
                    if isinstance(content, list):
                        return ' '.join(str(item) for item in content)
                    return str(content) if content else ""
                elif hasattr(choice, 'text'):
                    return str(choice.text)
            
            # Fallback to string conversion
            return str(response)
        except Exception as e:
            print(f"Error extracting response content: {e}")
            return f"Error processing response: {str(e)}"
    
    def generate_document_summary(self, document_info: Dict[str, Any]) -> str:
        """Generate a summary of the document using the LLM"""
        try:
            content_preview = document_info['content'][:2000]  # Limit content length
            
            prompt = f"""
            Analyze the following document and provide a comprehensive summary:
            
            File Name: {document_info['file_name']}
            File Type: {document_info['file_type']}
            
            Content Preview:
            {content_preview}
            
            Please provide:
            1. A brief overview of the document
            2. Key topics or themes identified
            3. If it contains data, describe the structure and main variables
            4. Any notable patterns or insights
            
            Keep the summary concise but informative.
            """
            
            return self._make_api_call_with_retry(prompt, max_tokens=500)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def perform_data_analysis(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis on structured data
        
        Args:
            df: DataFrame to analyze
            file_name: Name of the source file
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'summary_statistics': {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {},
            'correlations': None
        }
        
        # Summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis['summary_statistics'] = df[numeric_columns].describe().to_dict()
            
            # Correlation matrix for numeric columns
            if len(numeric_columns) > 1:
                analysis['correlations'] = df[numeric_columns].corr().to_dict()
        
        # Unique values for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            analysis['unique_values'][col] = df[col].nunique()
        
        # Store analysis results
        self.analysis_results[file_name] = analysis
        
        return analysis
    
    def create_visualizations(self, df: pd.DataFrame, file_name: str) -> List[str]:
        """
        Create various visualizations for the data
        
        Args:
            df: DataFrame to visualize
            file_name: Name of the source file
            
        Returns:
            List of paths to saved visualization files
        """
        visualization_paths = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Create output directory
        output_dir = f"visualizations_{file_name.replace('.', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Distribution plots for numeric columns
            if len(numeric_columns) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_columns[:4]):
                    if i < len(axes):
                        df[col].hist(bins=30, ax=axes[i])
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                
                plt.tight_layout()
                dist_path = os.path.join(output_dir, 'distributions.png')
                plt.savefig(dist_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(dist_path)
            
            # 2. Correlation heatmap
            if len(numeric_columns) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                corr_path = os.path.join(output_dir, 'correlation_heatmap.png')
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(corr_path)
            
            # 3. Box plots for numeric columns
            if len(numeric_columns) > 0:
                n_plots = min(3, len(numeric_columns))
                
                # Always create a figure with at least one subplot
                fig, ax_array = plt.subplots(1, n_plots, figsize=(15, 5), squeeze=False)
                axes = ax_array.flatten()
                
                for i, col in enumerate(numeric_columns[:n_plots]):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
                
                plt.tight_layout()
                box_path = os.path.join(output_dir, 'box_plots.png')
                plt.savefig(box_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(box_path)
            
            # 4. Bar charts for categorical columns
            if len(categorical_columns) > 0:
                n_cat_plots = min(2, len(categorical_columns))
                fig, ax_array = plt.subplots(1, n_cat_plots, figsize=(15, 6), squeeze=False)
                axes = ax_array.flatten()
                
                for i, col in enumerate(categorical_columns[:n_cat_plots]):
                    if df[col].nunique() <= 20:
                        value_counts = df[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'Top Values in {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Count')
                        axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                bar_path = os.path.join(output_dir, 'categorical_bars.png')
                plt.savefig(bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(bar_path)
                
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
        
        return visualization_paths
    
    def answer_question(self, question: str, context_files: Optional[List[str]] = None) -> str:
        """
        Answer a question based on the processed documents
        
        Args:
            question: User's question
            context_files: Specific files to use as context (if None, uses all files)
            
        Returns:
            Answer to the question
        """
        try:
            # Prepare context from documents
            context = ""
            
            if context_files is None:
                context_files = list(self.document_content.keys())
            
            for file_name in context_files:
                if file_name in self.document_content:
                    doc_info = self.document_content[file_name]
                    context += f"\n--- {file_name} ---\n"
                    context += f"File Type: {doc_info['file_type']}\n"
                    context += f"Summary: {doc_info['summary']}\n"
                    
                    # Add relevant content (truncated for API limits)
                    content_preview = doc_info['content'][:1500]
                    context += f"Content: {content_preview}\n"
                    
                    # Add analysis results if available
                    if file_name in self.analysis_results:
                        analysis = self.analysis_results[file_name]
                        context += f"Data Analysis Summary:\n"
                        context += f"Shape: {analysis['basic_info']['shape']}\n"
                        context += f"Columns: {analysis['basic_info']['columns']}\n"
                        if analysis['summary_statistics']:
                            context += f"Key Statistics Available: {list(analysis['summary_statistics'].keys())}\n"
            
            # Create the prompt
            prompt = f"""
            You are an intelligent data analyst. Based on the following document(s) and analysis, please answer the user's question accurately and comprehensively.
            
            CONTEXT FROM DOCUMENTS:
            {context[:4000]}  # Limit context length
            
            CONVERSATION HISTORY:
            {self._format_conversation_history()}
            
            USER QUESTION: {question}
            
            Please provide a detailed answer based on the available data and documents. If the question involves specific data analysis, calculations, or comparisons, please be precise and cite relevant statistics or findings from the documents.
            
            If you cannot answer the question based on the available information, please explain what additional information would be needed.
            """
            
            answer = self._make_api_call_with_retry(prompt, max_tokens=1000)
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'context_files': context_files
            })
            
            return answer
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted = ""
        for i, item in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
            formatted += f"Q{i+1}: {item['question']}\n"
            formatted += f"A{i+1}: {item['answer'][:200]}...\n\n"
        
        return formatted
    
    def generate_comprehensive_report(self, file_name: str) -> str:
        """
        Generate a comprehensive analysis report for a file
        
        Args:
            file_name: Name of the file to generate report for
            
        Returns:
            Comprehensive report string
        """
        if file_name not in self.document_content:
            return f"File {file_name} not found in processed documents."
        
        try:
            doc_info = self.document_content[file_name]
            report_prompt = f"""
            Generate a comprehensive analytical report for the following document:
            
            File: {file_name}
            Type: {doc_info['file_type']}
            Summary: {doc_info['summary']}
            
            Content Sample: {doc_info['content'][:2000]}
            
            Please provide:
            1. Executive Summary
            2. Key Findings
            3. Data Quality Assessment (if applicable)
            4. Trends and Patterns
            5. Recommendations
            6. Areas for Further Investigation
            
            Make the report professional and actionable.
            """
            
            return self._make_api_call_with_retry(report_prompt, max_tokens=1500)
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about all processed files"""
        info = {}
        for file_name, doc_info in self.document_content.items():
            info[file_name] = {
                'type': doc_info['file_type'],
                'has_data': file_name in self.data_frames,
                'summary': doc_info['summary'][:100] + "..." if len(doc_info['summary']) > 100 else doc_info['summary']
            }
        return info
    
    def _make_api_call_with_retry(self, prompt: str, max_tokens: int = 500, max_retries: int = 3) -> str:
        """Make API call with retry logic and exponential backoff"""
        # Use session settings if available
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                max_tokens = getattr(st.session_state, 'max_tokens', max_tokens)
                max_retries = getattr(st.session_state, 'max_retries', max_retries)
                temperature = getattr(st.session_state, 'temperature', 0.3)
            else:
                temperature = 0.3
        except:
            temperature = 0.3
        
        for attempt in range(max_retries):
            try:
                if not self.client:
                    return "API client not initialized properly"
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                
                return self._extract_response_content(response)
                
            except Exception as e:
                error_str = str(e)
                print(f"API call error (attempt {attempt + 1}/{max_retries}): {error_str}")
                
                if "rate limit" in error_str.lower() or "429" in error_str:
                    wait_time = (2 ** attempt) * 30  # Exponential backoff: 30s, 60s, 120s
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        return f"Rate limit exceeded. Please try again in a few minutes. Error: {error_str}"
                elif "authentication" in error_str.lower() or "unauthorized" in error_str.lower() or "401" in error_str:
                    return f"Authentication Error: Please check your API key. Error: {error_str}"
                elif "invalid" in error_str.lower() and "model" in error_str.lower():
                    return f"Model Error: The model '{self.model}' may not be available. Error: {error_str}"
                else:
                    # For other errors, wait a bit before retrying if we have retries left
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                        print(f"General error, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        return f"API Error: {error_str}"
        
        return "Maximum retries exceeded. Please try again later."
    
    def is_legal_document(self, content: str) -> bool:
        """Detect if document contains legal content"""
        legal_indicators = [
            'agreement', 'contract', 'party', 'whereas', 'plaintiff', 'defendant', 
            'court', 'statute', 'regulation', 'cfr', 'usc', 'whereas', 'breach',
            'liability', 'indemnification', 'termination', 'compliance', 'legal',
            'attorney', 'counsel', 'jurisdiction', 'patent', 'copyright', 'trademark'
        ]
        
        content_lower = content.lower()
        legal_term_count = sum(1 for term in legal_indicators if term in content_lower)
        
        # If more than 3 legal terms found, consider it a legal document
        return legal_term_count >= 3
    
    def detect_legal_document_type(self, content: str) -> str:
        """Detect the specific type of legal document"""
        return self.legal_analyzer.classify_legal_document(content)
    
    def perform_legal_analysis(self, file_name: str) -> Dict[str, Any]:
        """Perform comprehensive legal analysis on a document"""
        if file_name not in self.document_content:
            return {"error": "Document not found"}
        
        content = self.document_content[file_name]['content']
        return self.legal_analyzer.analyze_legal_document(content)
    
    def compare_legal_documents(self, file1: str, file2: str) -> Dict[str, Any]:
        """Compare two legal documents for differences"""
        if file1 not in self.document_content or file2 not in self.document_content:
            return {"error": "One or both documents not found"}
        
        content1 = self.document_content[file1]['content']
        content2 = self.document_content[file2]['content']
        
        return self.legal_analyzer.compare_legal_documents(content1, content2)
    
    def extract_legal_summary(self, file_name: str) -> str:
        """Generate a legal-focused summary"""
        if file_name not in self.legal_analysis_results:
            return "No legal analysis available for this document"
        
        analysis = self.legal_analysis_results[file_name]
        
        summary_parts = []
        summary_parts.append(f"Document Type: {analysis.get('document_type', 'Unknown')}")
        
        if analysis.get('risk_assessment'):
            risk = analysis['risk_assessment']
            summary_parts.append(f"Risk Level: {risk.get('overall_risk', 'Unknown')}")
        
        if analysis.get('entities'):
            entities = analysis['entities'][:3]  # Top 3 entities
            entity_names = [e.name for e in entities]
            summary_parts.append(f"Key Parties: {', '.join(entity_names)}")
        
        if analysis.get('key_obligations'):
            obligations = analysis['key_obligations'][:2]  # Top 2 obligations
            summary_parts.append(f"Key Obligations: {'; '.join(obligations)}")
        
        return '\n'.join(summary_parts)
    
    def create_legal_visualizations(self, file_name: str) -> List[str]:
        """Create legal-specific visualizations"""
        if file_name not in self.legal_analysis_results:
            return []
        
        analysis = self.legal_analysis_results[file_name]
        visualization_paths = []
        
        try:
            output_dir = f"legal_visualizations_{file_name.replace('.', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Risk Assessment Chart
            if analysis.get('risk_assessment'):
                risk_data = analysis['risk_assessment']
                risk_factors = risk_data.get('risk_factors', [])
                
                if risk_factors:
                    risk_levels = [factor['level'] for factor in risk_factors]
                    risk_counts = {'high': 0, 'medium': 0, 'low': 0}
                    
                    for level in risk_levels:
                        risk_counts[level] += 1
                    
                    plt.figure(figsize=(10, 6))
                    colors = ['red', 'orange', 'green']
                    plt.bar(list(risk_counts.keys()), list(risk_counts.values()), color=colors)
                    plt.title('Legal Risk Assessment')
                    plt.xlabel('Risk Level')
                    plt.ylabel('Number of Risk Factors')
                    
                    risk_path = os.path.join(output_dir, 'risk_assessment.png')
                    plt.savefig(risk_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths.append(risk_path)
            
            # 2. Entity Relationship Chart
            if analysis.get('entities'):
                entities = analysis['entities']
                entity_types = {}
                
                for entity in entities:
                    entity_type = entity.entity_type
                    if entity_type in entity_types:
                        entity_types[entity_type] += 1
                    else:
                        entity_types[entity_type] = 1
                
                if entity_types:
                    plt.figure(figsize=(8, 8))
                    plt.pie(list(entity_types.values()), labels=list(entity_types.keys()), autopct='%1.1f%%')
                    plt.title('Legal Entities Distribution')
                    
                    entities_path = os.path.join(output_dir, 'entities_distribution.png')
                    plt.savefig(entities_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths.append(entities_path)
            
            # 3. Compliance Score Gauge
            if analysis.get('compliance_check'):
                compliance = analysis['compliance_check']
                score = compliance.get('score', 0)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create gauge chart
                theta = np.linspace(0, np.pi, 100)
                r = np.ones_like(theta)
                
                ax = plt.subplot(111, polar=True)
                ax.plot(theta, r, color='lightgray', linewidth=10)
                
                # Color based on score
                if score >= 80:
                    color = 'green'
                elif score >= 60:
                    color = 'orange'
                else:
                    color = 'red'
                
                score_theta = np.linspace(0, np.pi * (score / 100), 50)
                ax.plot(score_theta, np.ones_like(score_theta), color=color, linewidth=10)
                
                ax.set_ylim(0, 1.2)
                ax.set_title(f'Compliance Score: {score}%', pad=20)
                ax.set_ylim(0, 1.2)
                ax.grid(True)
                
                compliance_path = os.path.join(output_dir, 'compliance_score.png')
                plt.savefig(compliance_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(compliance_path)
            
            # 4. Timeline Visualization
            if analysis.get('dates'):
                dates = analysis['dates']
                
                if dates:
                    plt.figure(figsize=(12, 6))
                    
                    date_types = [date.date_type for date in dates]
                    importance_levels = [date.importance for date in dates]
                    
                    # Create timeline
                    y_pos = range(len(dates))
                    colors = ['red' if imp == 'critical' else 'orange' if imp == 'important' else 'blue' 
                             for imp in importance_levels]
                    
                    plt.barh(y_pos, [1] * len(dates), color=colors)
                    plt.yticks(y_pos, [f"{date.date_type}: {date.date_text}" for date in dates])
                    plt.xlabel('Timeline')
                    plt.title('Legal Dates and Deadlines')
                    
                    timeline_path = os.path.join(output_dir, 'legal_timeline.png')
                    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths.append(timeline_path)
            
        except Exception as e:
            print(f"Error creating legal visualizations: {str(e)}")
        
        return visualization_paths
    
    def generate_legal_report(self, file_name: str) -> str:
        """Generate comprehensive legal analysis report"""
        if file_name not in self.legal_analysis_results:
            return "No legal analysis available for this document"
        
        analysis = self.legal_analysis_results[file_name]
        
        try:
            report_prompt = f"""
            Generate a comprehensive legal analysis report based on the following analysis:
            
            Document: {file_name}
            Document Type: {analysis.get('document_type', 'Unknown')}
            
            Risk Assessment: {analysis.get('risk_assessment', {})}
            Entities Found: {len(analysis.get('entities', []))}
            Citations Found: {len(analysis.get('citations', []))}
            Legal Clauses: {len(analysis.get('clauses', []))}
            Key Obligations: {analysis.get('key_obligations', [])}
            Compliance Score: {analysis.get('compliance_check', {}).get('score', 'N/A')}
            
            Plain English Summary: {analysis.get('plain_english_summary', '')}
            
            Please provide:
            1. Executive Summary
            2. Legal Risk Analysis
            3. Key Findings and Recommendations
            4. Compliance Assessment
            5. Action Items and Next Steps
            
            Make the report suitable for legal professionals and decision makers.
            """
            
            return self._make_api_call_with_retry(report_prompt, max_tokens=2000)
            
        except Exception as e:
            return f"Error generating legal report: {str(e)}"


class LegalSecurityManager:
    """Handle security and privacy for legal documents"""
    
    def __init__(self):
        self.access_logs = []
        self.privilege_markers = [
            'attorney-client privilege', 'privileged and confidential', 
            'work product', 'attorney work product', 'confidential',
            'privileged communication'
        ]
    
    def check_privilege(self, content: str) -> Dict[str, Any]:
        """Check for attorney-client privilege indicators"""
        content_lower = content.lower()
        privilege_found = []
        
        for marker in self.privilege_markers:
            if marker in content_lower:
                privilege_found.append(marker)
        
        return {
            'is_privileged': len(privilege_found) > 0,
            'privilege_markers': privilege_found,
            'warning': 'This document may contain privileged information' if privilege_found else None
        }
    
    def log_access(self, user_id: str, document_name: str, action: str):
        """Log document access for audit trail"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'document': document_name,
            'action': action,
            'hash': hashlib.md5(f"{timestamp}{user_id}{document_name}".encode()).hexdigest()[:8]
        }
        self.access_logs.append(log_entry)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Basic encryption for sensitive legal data (placeholder)"""
        # In production, use proper encryption libraries
        encoded = base64.b64encode(data.encode()).decode()
        return f"ENCRYPTED:{encoded}"
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Basic decryption for sensitive legal data (placeholder)"""
        if encrypted_data.startswith("ENCRYPTED:"):
            encoded = encrypted_data[10:]
            return base64.b64decode(encoded).decode()
        return encrypted_data
    
    def generate_confidentiality_notice(self) -> str:
        """Generate a confidentiality notice for legal documents"""
        return """
        CONFIDENTIALITY NOTICE:
        This document may contain attorney-client privileged information and/or 
        attorney work product. If you are not the intended recipient, please 
        notify the sender immediately and delete this document. Any unauthorized 
        review, use, disclosure or distribution is prohibited.
        """


# Legal Document Analysis Classes
class LegalDocumentType(Enum):
    CONTRACT = "contract"
    STATUTE = "statute" 
    CASE_LAW = "case_law"
    REGULATION = "regulation"
    PATENT = "patent"
    LEGAL_BRIEF = "legal_brief"
    GENERAL = "general"

@dataclass
class LegalEntity:
    name: str
    entity_type: str  # person, organization, court, jurisdiction
    role: str  # plaintiff, defendant, party, judge, etc.
    confidence: float

@dataclass
class LegalCitation:
    text: str
    case_name: str = ""
    court: str = ""
    year: str = ""
    volume: str = ""
    reporter: str = ""
    page: str = ""
    citation_type: str = ""  # case, statute, regulation
    confidence: float = 0.0

@dataclass
class LegalClause:
    text: str
    clause_type: str  # termination, payment, liability, etc.
    risk_level: str  # low, medium, high
    obligations: List[str]
    section: str = ""

@dataclass
class LegalDate:
    date_text: str
    parsed_date: datetime
    date_type: str  # deadline, effective_date, expiration, etc.
    importance: str  # critical, important, informational

class LegalDocumentAnalyzer:
    """Advanced legal document analysis with specialized features"""
    
    def __init__(self, api_key: str):
        # Initialize OpenRouter client for legal analysis
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.getenv('OPENROUTER_APP_URL', 'https://github.com/DevanshSrajput/Dataa_Analyst_Agent'),
                "X-Title": os.getenv('OPENROUTER_APP_NAME', 'AI Document Analyst v2.0')
            }
        )
        self.model = os.getenv('DEFAULT_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
        
        # Legal patterns and dictionaries
        self.legal_terms = {
            'contract_terms': ['whereas', 'party', 'agreement', 'consideration', 'breach', 'termination', 'liability', 'indemnification'],
            'court_terms': ['plaintiff', 'defendant', 'court', 'judge', 'jury', 'verdict', 'appeal', 'motion'],
            'regulatory_terms': ['regulation', 'compliance', 'statute', 'code', 'section', 'subsection', 'amendment']
        }
        
        # Citation patterns
        self.citation_patterns = {
            'case_citation': r'(\d+)\s+([A-Za-z.]+)\s+(\d+)(?:\s*\(([^)]+)\s+(\d{4})\))?',
            'statute_citation': r'(\d+)\s+U\.?S\.?C\.?\s+§?\s*(\d+)',
            'code_citation': r'(\d+)\s+C\.?F\.?R\.?\s+§?\s*(\d+)'
        }
        
        # Risk keywords
        self.risk_keywords = {
            'high': ['unlimited liability', 'personal guarantee', 'penalty', 'forfeiture', 'criminal'],
            'medium': ['liquidated damages', 'indemnification', 'non-compete', 'confidentiality breach'],
            'low': ['notice required', 'reasonable efforts', 'best practices', 'standard terms']
        }

    def analyze_legal_document(self, content: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive legal document analysis"""
        try:
            # Determine document type if not provided
            if not document_type:
                document_type = self.classify_legal_document(content)
            
            analysis = {
                'document_type': document_type,
                'entities': self.extract_legal_entities(content),
                'citations': self.extract_legal_citations(content),
                'clauses': self.extract_legal_clauses(content),
                'dates': self.extract_legal_dates(content),
                'risk_assessment': self.assess_legal_risks(content),
                'plain_english_summary': self.translate_to_plain_english(content),
                'compliance_check': self.check_compliance(content, document_type),
                'key_obligations': self.extract_obligations(content)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f"Legal analysis failed: {str(e)}"}

    def classify_legal_document(self, content: str) -> str:
        """Classify the type of legal document"""
        content_lower = content.lower()
        
        # Contract indicators
        if any(term in content_lower for term in ['agreement', 'contract', 'party', 'whereas', 'consideration']):
            return LegalDocumentType.CONTRACT.value
        
        # Case law indicators
        elif any(term in content_lower for term in ['plaintiff', 'defendant', 'court', 'appeal', 'judgment']):
            return LegalDocumentType.CASE_LAW.value
            
        # Statute indicators
        elif any(term in content_lower for term in ['statute', 'code', 'section', 'subsection', 'act']):
            return LegalDocumentType.STATUTE.value
            
        # Regulation indicators
        elif any(term in content_lower for term in ['regulation', 'cfr', 'federal register', 'rule']):
            return LegalDocumentType.REGULATION.value
            
        # Patent indicators
        elif any(term in content_lower for term in ['patent', 'invention', 'claim', 'prior art', 'embodiment']):
            return LegalDocumentType.PATENT.value
            
        else:
            return LegalDocumentType.GENERAL.value

    def extract_legal_entities(self, content: str) -> List[LegalEntity]:
        """Extract legal entities like parties, courts, judges"""
        entities = []
        
        # Simple pattern-based extraction (would be enhanced with NLP)
        patterns = {
            'party': r'(?:party|parties)\s+([A-Z][A-Za-z\s]+)(?=\s|,|\.)',
            'court': r'(?:court|Court)\s+([A-Z][A-Za-z\s]+)(?=\s|,|\.)',
            'judge': r'(?:judge|Judge|justice|Justice)\s+([A-Z][A-Za-z\s]+)(?=\s|,|\.)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for match in matches[:5]:  # Limit to avoid noise
                entities.append(LegalEntity(
                    name=match.strip(),
                    entity_type=entity_type,
                    role=entity_type,
                    confidence=0.8
                ))
        
        return entities

    def extract_legal_citations(self, content: str) -> List[LegalCitation]:
        """Extract and parse legal citations"""
        citations = []
        
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                citation = LegalCitation(
                    text=match.group(0),
                    citation_type=citation_type,
                    confidence=0.9
                )
                
                if citation_type == 'case_citation':
                    citation.volume = match.group(1) if match.group(1) else ""
                    citation.reporter = match.group(2) if match.group(2) else ""
                    citation.page = match.group(3) if match.group(3) else ""
                    citation.court = match.group(4) if match.group(4) else ""
                    citation.year = match.group(5) if match.group(5) else ""
                
                citations.append(citation)
        
        return citations

    def extract_legal_clauses(self, content: str) -> List[LegalClause]:
        """Extract and analyze legal clauses"""
        clauses = []
        
        # Split content into potential clauses
        sections = re.split(r'\n\s*\n|\. [A-Z]', content)
        
        clause_types = {
            'termination': ['termination', 'terminate', 'end', 'expire'],
            'liability': ['liability', 'liable', 'responsible', 'damages'],
            'payment': ['payment', 'pay', 'fee', 'cost', 'price'],
            'confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'trade secret'],
            'indemnification': ['indemnify', 'indemnification', 'hold harmless', 'defend']
        }
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            for clause_type, keywords in clause_types.items():
                if any(keyword in section.lower() for keyword in keywords):
                    risk_level = self.assess_clause_risk(section)
                    obligations = self.extract_clause_obligations(section)
                    
                    clauses.append(LegalClause(
                        text=section.strip()[:500],  # Limit length
                        clause_type=clause_type,
                        risk_level=risk_level,
                        obligations=obligations
                    ))
                    break
        
        return clauses

    def extract_legal_dates(self, content: str) -> List[LegalDate]:
        """Extract important legal dates and deadlines"""
        dates = []
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})',
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})'
        ]
        
        date_types = {
            'deadline': ['deadline', 'due', 'expire', 'end', 'final'],
            'effective_date': ['effective', 'commence', 'begin', 'start'],
            'expiration': ['expiration', 'expire', 'end', 'terminate']
        }
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                date_text = match.group(1)
                try:
                    # Simple date parsing (would use more robust parsing in production)
                    parsed_date = datetime.now()  # Placeholder
                    
                    # Determine date type based on context
                    context = content[max(0, match.start()-50):match.end()+50].lower()
                    date_type = 'general'
                    importance = 'informational'
                    
                    for dtype, keywords in date_types.items():
                        if any(keyword in context for keyword in keywords):
                            date_type = dtype
                            importance = 'critical' if dtype == 'deadline' else 'important'
                            break
                    
                    dates.append(LegalDate(
                        date_text=date_text,
                        parsed_date=parsed_date,
                        date_type=date_type,
                        importance=importance
                    ))
                except:
                    continue
        
        return dates[:10]  # Limit results

    def assess_legal_risks(self, content: str) -> Dict[str, Any]:
        """Assess legal risks in the document"""
        risk_assessment = {
            'overall_risk': 'low',
            'risk_factors': [],
            'recommendations': []
        }
        
        content_lower = content.lower()
        high_risk_count = 0
        medium_risk_count = 0
        
        # Check for risk keywords
        for risk_level, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    risk_assessment['risk_factors'].append({
                        'factor': keyword,
                        'level': risk_level,
                        'context': self.extract_context(content, keyword)
                    })
                    
                    if risk_level == 'high':
                        high_risk_count += 1
                    elif risk_level == 'medium':
                        medium_risk_count += 1
        
        # Determine overall risk
        if high_risk_count > 0:
            risk_assessment['overall_risk'] = 'high'
        elif medium_risk_count > 1:
            risk_assessment['overall_risk'] = 'medium'
        
        # Generate recommendations
        if high_risk_count > 0:
            risk_assessment['recommendations'].append("Consider legal review for high-risk clauses")
        if medium_risk_count > 0:
            risk_assessment['recommendations'].append("Review medium-risk terms carefully")
        
        return risk_assessment

    def translate_to_plain_english(self, content: str) -> str:
        """Translate legal jargon to plain English"""
        try:
            prompt = f"""
            Translate the following legal text into plain English that a non-lawyer can understand.
            Explain legal terms and concepts in simple language while maintaining accuracy.
            
            Legal Text:
            {content[:2000]}
            
            Provide a clear, concise explanation in plain English:
            """
            
            return self._make_api_call(prompt, max_tokens=800)
        except Exception as e:
            return f"Translation failed: {str(e)}"

    def check_compliance(self, content: str, document_type: str) -> Dict[str, Any]:
        """Check compliance requirements and gaps"""
        compliance = {
            'requirements': [],
            'gaps': [],
            'score': 0
        }
        
        # Basic compliance checks based on document type
        if document_type == LegalDocumentType.CONTRACT.value:
            required_elements = ['consideration', 'offer', 'acceptance', 'parties']
            found_elements = []
            
            for element in required_elements:
                if element in content.lower():
                    found_elements.append(element)
                else:
                    compliance['gaps'].append(f"Missing: {element}")
            
            compliance['score'] = len(found_elements) / len(required_elements) * 100
            compliance['requirements'] = required_elements
        
        return compliance

    def extract_obligations(self, content: str) -> List[str]:
        """Extract key obligations from legal text"""
        obligations = []
        
        obligation_patterns = [
            r'shall\s+([^.]+)',
            r'must\s+([^.]+)',
            r'required\s+to\s+([^.]+)',
            r'agrees\s+to\s+([^.]+)'
        ]
        
        for pattern in obligation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                obligation = match.strip()
                if len(obligation) > 10 and len(obligation) < 200:
                    obligations.append(obligation)
        
        return obligations[:10]  # Limit results

    def assess_clause_risk(self, clause: str) -> str:
        """Assess risk level of a specific clause"""
        clause_lower = clause.lower()
        
        for risk_level, keywords in self.risk_keywords.items():
            if any(keyword in clause_lower for keyword in keywords):
                return risk_level
        
        return 'low'

    def extract_clause_obligations(self, clause: str) -> List[str]:
        """Extract obligations from a specific clause"""
        return self.extract_obligations(clause)

    def extract_context(self, content: str, keyword: str) -> str:
        """Extract context around a keyword"""
        index = content.lower().find(keyword.lower())
        if index != -1:
            start = max(0, index - 50)
            end = min(len(content), index + len(keyword) + 50)
            return content[start:end].strip()
        return ""

    def compare_legal_documents(self, doc1_content: str, doc2_content: str) -> Dict[str, Any]:
        """Compare two legal documents (redlining functionality)"""
        try:
            # Simple diff comparison
            diff = list(difflib.unified_diff(
                doc1_content.splitlines(keepends=True),
                doc2_content.splitlines(keepends=True),
                fromfile='Document 1',
                tofile='Document 2'
            ))
            
            added_lines = []
            removed_lines = []
            
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines.append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    removed_lines.append(line[1:])
            
            return {
                'differences_found': len(added_lines) + len(removed_lines) > 0,
                'added_content': added_lines[:20],  # Limit results
                'removed_content': removed_lines[:20],
                'similarity_score': self.calculate_similarity(doc1_content, doc2_content)
            }
        except Exception as e:
            return {'error': f"Document comparison failed: {str(e)}"}

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        # Simple character-based similarity
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return round(matcher.ratio() * 100, 2)

    def _make_api_call(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call for legal analysis"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
                stream=False
            )
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return str(choice.message.content)
            
            return "Analysis completed but response format unclear"
        except Exception as e:
            return f"API call failed: {str(e)}"


def create_streamlit_ui():
    """Create the Streamlit user interface"""
    # Page configuration with custom styling
    st.set_page_config(
        page_title="📊 AI Document Analyst | Smart Document Processing",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/ARCH_USERS/Dataa_Analyst_Agent',
            'Report a bug': "mailto:support@archusers.com",
            'About': "AI-powered document analysis tool by ARCH_USERS"
        }
    )
    
    # Initialize theme if not set
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    
    # Custom CSS for modern styling with theme support
    theme_mode = st.session_state.theme_mode
    
    if theme_mode == 'dark':
        css_theme = """
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-tertiary: #3d3d3d;
            --text-primary: #fafafa;
            --text-secondary: #e0e0e0;
            --accent-color: #667eea;
            --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            --border-color: #555555;
            --card-bg: #262730;
            --upload-bg: #1e1e2e;
            --tab-bg: #262730;
            --tab-selected: #667eea;
        }
        
        .stApp {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Sidebar styling for dark mode - comprehensive */
        .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-k1vhr4, .css-18e3th9 {
            background-color: var(--bg-secondary) !important;
        }
        
        /* Sidebar - force all text elements to be visible */
        .css-1d391kg, 
        .css-1d391kg *,
        .css-1d391kg div,
        .css-1d391kg span,
        .css-1d391kg p,
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
        .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6,
        .css-1d391kg label,
        .css-1d391kg li,
        .css-1d391kg .stMarkdown,
        .css-1d391kg .stMarkdown *,
        .css-1d391kg .stText,
        .css-1d391kg .element-container,
        .css-1d391kg .element-container *,
        .css-1d391kg .block-container,
        .css-1d391kg .block-container * {
            color: var(--text-primary) !important;
            background-color: var(--bg-secondary) !important;
        }
        
        /* Sidebar specific elements */
        .css-1d391kg .stMetric,
        .css-1d391kg .stMetric *,
        .css-1d391kg [data-testid="metric-container"],
        .css-1d391kg [data-testid="metric-container"] * {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Sidebar buttons */
        .css-1d391kg .stButton > button {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .css-1d391kg .stButton > button:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        /* Sidebar metrics */
        .css-1d391kg .metric-container,
        .css-1d391kg [data-testid="metric-container"] {
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }
        
        /* Sidebar expander */
        .css-1d391kg .streamlit-expanderHeader,
        .css-1d391kg .streamlit-expanderContent {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Main content area */
        .main .block-container {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Force all text elements to be visible in dark mode */
        h1, h2, h3, h4, h5, h6, p, div, span, label, li, strong, em, a {
            color: var(--text-primary) !important;
        }
        
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span, .stMarkdown * {
            color: var(--text-primary) !important;
        }
        
        /* Sidebar specific overrides */
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] strong {
            color: var(--text-primary) !important;
            background-color: var(--bg-secondary) !important;
        }
        
        /* Input elements */
        .stSelectbox > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stSelectbox > div > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Dropdown options */
        .stSelectbox > div > div > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Selectbox dropdown menu - comprehensive styling */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Selectbox dropdown container */
        .stSelectbox [data-baseweb="popover"] {
            background-color: var(--bg-secondary) !important;
        }
        
        /* Selectbox dropdown list */
        .stSelectbox [data-baseweb="menu"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Selectbox option items */
        .stSelectbox [role="option"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        .stSelectbox [role="option"]:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        /* Selectbox selected value */
        .stSelectbox [data-baseweb="select"] [data-baseweb="input"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Selectbox label */
        .stSelectbox label {
            color: var(--text-primary) !important;
            font-weight: bold !important;
        }
        
        /* Additional selectbox styling for dropdown options */
        .stSelectbox div[data-baseweb="select"] ul li {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        .stSelectbox div[data-baseweb="select"] ul li:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        /* Fix for dropdown text visibility */
        .stSelectbox [data-testid="stSelectbox"] > div > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Dropdown menu items styling */
        .stSelectbox [data-testid="stSelectbox"] [role="listbox"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stSelectbox [data-testid="stSelectbox"] [role="listbox"] [role="option"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            padding: 8px 12px !important;
        }
        
        .stSelectbox [data-testid="stSelectbox"] [role="listbox"] [role="option"]:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        .stTextInput > div > div > input {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
        
        .stTextInput label {
            color: var(--text-primary) !important;
        }
        
        .stTextArea > div > div > textarea {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
        
        .stTextArea label {
            color: var(--text-primary) !important;
        }
        
        /* Metrics and info boxes */
        .stMetric {
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }
        
        .stInfo, .stSuccess, .stWarning, .stError {
            color: var(--text-primary) !important;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        .streamlit-expanderContent {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* DataFrame */
        .stDataFrame {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Progress bar container */
        .stProgress > div > div {
            background-color: var(--bg-secondary) !important;
        }
        
        /* Additional emergency dropdown fix for dark mode */
        .stSelectbox,
        .stSelectbox *,
        .stSelectbox div,
        .stSelectbox span,
        .stSelectbox p,
        .stSelectbox label,
        [data-testid="stSelectbox"],
        [data-testid="stSelectbox"] *,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] span {
            color: var(--text-primary) !important;
        }
        
        /* Force dropdown background and text for all possible selectors */
        .stSelectbox > div,
        .stSelectbox > div > div,
        .stSelectbox > div > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        """
    else:
        css_theme = """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --accent-color: #667eea;
            --accent-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            --border-color: #dee2e6;
            --card-bg: #f8f9fa;
            --upload-bg: #f8f9fa;
            --tab-bg: #f0f2f6;
            --tab-selected: #667eea;
        }
        
        .stApp {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Sidebar styling for light mode */
        .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-k1vhr4, .css-18e3th9 {
            background-color: var(--bg-secondary) !important;
        }
        
        /* Sidebar container and all child elements */
        .css-1d391kg, .css-1d391kg * {
            color: var(--text-primary) !important;
        }
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: var(--text-primary) !important;
        }
        
        .stMarkdown, .stMarkdown p, .stMarkdown div {
            color: var(--text-primary) !important;
        }
        
        /* Light mode selectbox styling */
        .stSelectbox > div > div {
            background-color: white !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stSelectbox label {
            color: var(--text-primary) !important;
            font-weight: bold !important;
        }
        
        /* Light mode dropdown options */
        .stSelectbox div[role="listbox"] {
            background-color: white !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stSelectbox div[role="option"] {
            background-color: white !important;
            color: var(--text-primary) !important;
        }
        
        .stSelectbox div[role="option"]:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        /* Light mode buttons */
        .stButton > button {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stButton > button:hover {
            background-color: var(--accent-color) !important;
            color: white !important;
        }
        
        /* Light mode metrics */
        .css-1d391kg .stMetric,
        .css-1d391kg .stMetric *,
        .css-1d391kg [data-testid="metric-container"],
        .css-1d391kg [data-testid="metric-container"] * {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Light mode file uploader */
        .stFileUploader > div {
            background-color: var(--upload-bg) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 10px !important;
        }
        
        /* Light mode text inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: white !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Light mode tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--tab-bg) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-primary) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--tab-selected) !important;
            color: white !important;
        }
        
        /* Light mode dataframe */
        .stDataFrame {
            background-color: white !important;
            color: var(--text-primary) !important;
        }
        
        /* Light mode expander */
        .streamlit-expanderHeader,
        .streamlit-expanderContent {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }
        """
    
    st.markdown(f"""
    <style>
    {css_theme}
    
    /* Main styling */
    .main-header {{
        background: var(--accent-gradient);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .feature-card {{
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--accent-color);
        margin: 1rem 0;
        color: var(--text-primary);
    }}
    
    .metric-card {{
        background: var(--accent-gradient);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }}
    
    .upload-zone {{
        border: 2px dashed var(--accent-color);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: var(--upload-bg);
        margin: 1rem 0;
        color: var(--text-primary);
    }}
    
    .chat-container {{
        background: var(--card-bg);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: var(--text-primary);
    }}
    
    .theme-toggle {{
        background: var(--accent-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        margin: 0.5rem;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: var(--tab-bg) !important;
        border-radius: 10px 10px 0 0;
        color: var(--text-primary) !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--tab-selected) !important;
        color: white !important;
    }}
    
    /* Enhanced Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Universal selectbox styling for all themes */
    .stSelectbox {{
        color: var(--text-primary) !important;
    }}
    
    .stSelectbox > div {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    .stSelectbox > div > div {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Dropdown options styling */
    .stSelectbox div[role="listbox"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }}
    
    .stSelectbox div[role="option"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 8px 12px !important;
    }}
    
    .stSelectbox div[role="option"]:hover {{
        background-color: var(--accent-color) !important;
        color: white !important;
    }}
    
    /* Force visibility for all selectbox text */
    .stSelectbox * {{
        color: var(--text-primary) !important;
    }}
    
    /* Additional comprehensive selectbox styling */
    .stSelectbox [data-baseweb="select"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Dropdown popover styling */
    [data-baseweb="popover"] .stSelectbox,
    [data-baseweb="popover"] [data-baseweb="menu"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Ensure all dropdown text is visible */
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="menu"] div,
    .stSelectbox [data-baseweb="menu"] span,
    .stSelectbox [role="option"] span,
    .stSelectbox [role="option"] div {{
        color: var(--text-primary) !important;
        background-color: var(--bg-secondary) !important;
    }}
    
    /* Fix for dropdown arrow and controls */
    .stSelectbox [data-baseweb="select"] svg {{
        fill: var(--text-primary) !important;
    }}
    
    /* Comprehensive option styling */
    .stSelectbox [role="option"],
    .stSelectbox [data-baseweb="menu"] li,
    .stSelectbox ul li {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 8px 12px !important;
    }}
    
    .stSelectbox [role="option"]:hover,
    .stSelectbox [data-baseweb="menu"] li:hover,
    .stSelectbox ul li:hover {{
        background-color: var(--accent-color) !important;
        color: white !important;
    }}
    
    /* Sidebar text elements */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6,
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span,
    .css-1d391kg label, .css-1d391kg .stMarkdown {{
        color: var(--text-primary) !important;
    }}
    
    /* Sidebar buttons */
    .css-1d391kg .stButton > button {{
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }}
    
    /* Sidebar metrics */
    .css-1d391kg .metric-container {{
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }}
    
    /* General text improvements */
    .stText, .stCaption, .stCode {{
        color: var(--text-primary) !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        color: var(--text-primary) !important;
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
    }}
    
    .stButton > button:hover {{
        color: var(--text-primary) !important;
        border-color: var(--accent-color) !important;
    }}
    
    /* Slider styling */
    .stSlider {{
        color: var(--text-primary) !important;
    }}
    
    .stSlider label {{
        color: var(--text-primary) !important;
    }}
    
    /* Hide default streamlit styling */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Additional dark mode support */
    .stContainer {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Fix for spinner and loading elements */
    .stSpinner {{
        color: var(--text-primary) !important;
    }}
    
    /* Fix for code blocks */
    .stCodeBlock {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Fix for alerts and messages */
    .stAlert {{
        color: var(--text-primary) !important;
    }}
    
    /* Fix for columns */
    .css-ocqkz7 {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Fix for expander headers in dark mode */
    .streamlit-expanderHeader p {{
        color: var(--text-primary) !important;
    }}
    
    .streamlit-expanderContent .stMarkdown {{
        color: var(--text-primary) !important;
    }}
    
    /* Additional BaseWeb dropdown fixes */
    [data-baseweb="select"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    [data-baseweb="menu"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }}
    
    [data-baseweb="menu"] [role="option"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
    }}
    
    [data-baseweb="menu"] [role="option"]:hover {{
        background-color: var(--accent-color) !important;
        color: white !important;
    }}
    
    /* Dropdown input field */
    [data-baseweb="select"] input {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Selected value display */
    [data-baseweb="select"] [data-baseweb="input"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Dropdown container */
    [data-baseweb="popover"] {{
        background-color: var(--bg-secondary) !important;
    }}
    
    /* Fix any remaining invisible text */
    .stSelectbox span,
    .stSelectbox div,
    [data-baseweb="select"] span,
    [data-baseweb="menu"] span {{
        color: var(--text-primary) !important;
    }}
    
    /* Ultra-aggressive dropdown text fix */
    .stSelectbox * {{
        color: var(--text-primary) !important;
    }}
    
    /* Force all dropdown elements to be visible */
    div[role="listbox"],
    div[role="listbox"] *,
    div[role="option"],
    div[role="option"] *,
    [data-baseweb="menu"],
    [data-baseweb="menu"] *,
    [data-baseweb="select"],
    [data-baseweb="select"] * {{
        color: var(--text-primary) !important;
        background-color: var(--bg-secondary) !important;
    }}
    
    /* Specific fix for Streamlit selectbox in all states */
    .stSelectbox [data-value],
    .stSelectbox [data-value] *,
    .stSelectbox .st-emotion-cache-1p0byqe,
    .stSelectbox .st-emotion-cache-1p0byqe * {{
        color: var(--text-primary) !important;
        background-color: var(--bg-secondary) !important;
    }}
    
    /* Override any inherited text colors */
    .stSelectbox > div > div > div,
    .stSelectbox > div > div > div *,
    .stSelectbox ul,
    .stSelectbox ul *,
    .stSelectbox li,
    .stSelectbox li * {{
        color: var(--text-primary) !important;
    }}
    
    /* Final fallback for any missed elements */
    [data-testid*="selectbox"] *,
    [class*="selectbox"] *,
    [class*="dropdown"] *,
    [class*="menu"] * {{
        color: var(--text-primary) !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Modern Header with gradient and theme indicator
    theme_indicator = "🌙" if st.session_state.theme_mode == 'dark' else "☀️"
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI Document Analyst {theme_indicator}</h1>
        <p style="font-size: 1.2em; margin: 0;">Transform your documents into actionable insights with AI</p>
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Built by ARCH_USERS | Powered by OpenRouter AI | {st.session_state.theme_mode.title()} Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent with enhanced error handling
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Check for temporary API key override
    if 'temp_api_key' in st.session_state and st.session_state.temp_api_key:
        api_key = st.session_state.temp_api_key
    
    if not api_key:
        st.error("🔐 **API Key Required!** Please set your OPENROUTER_API_KEY or enter it below")
        
        # Provide temporary API key input
        col1, col2 = st.columns([3, 1])
        with col1:
            temp_key = st.text_input(
                "Enter your OpenRouter API Key:",
                type="password",
                placeholder="sk-or-v1-...",
                help="Get your API key from https://openrouter.ai/"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🔑 Use Key"):
                if temp_key:
                    # Test the key first
                    with st.spinner("Testing API key..."):
                        is_valid, message = test_api_key(temp_key)
                        if is_valid:
                            st.session_state.temp_api_key = temp_key
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        
        with st.expander("🔧 How to get an API Key"):
            st.markdown("""
            1. **Visit [OpenRouter](https://openrouter.ai/)**
            2. **Sign up or log in** to your account
            3. **Go to API Keys** section in your dashboard
            4. **Create a new API key**
            5. **Copy and paste it above** or add to your `.env` file
            """)
        
        if temp_key:
            st.info("💡 Click 'Use Key' button to test your API key")
        return
    
    if 'agent' not in st.session_state or 'agent_error' in st.session_state:
        with st.spinner("🚀 Initializing AI Agent..."):
            try:
                st.session_state.agent = DocumentAnalystAgent(api_key)
                st.success("✅ AI Agent ready for action!")
                # Clear any previous errors
                if 'agent_error' in st.session_state:
                    del st.session_state.agent_error
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                
                # Check if it's an API key error
                if "401" in str(e) or "User not found" in str(e) or "No auth credentials" in str(e):
                    st.session_state.agent_error = True
                    st.warning("🔑 **API Key Issue Detected!** Your current API key appears to be invalid or expired.")
                    
                    # Provide option to enter new key
                    st.markdown("### 🔄 **Try a New API Key:**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_api_key = st.text_input(
                            "Enter a new OpenRouter API Key:",
                            type="password",
                            placeholder="sk-or-v1-...",
                            key="new_api_key_input"
                        )
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Try New Key"):
                            if new_api_key:
                                # Test the new key first
                                with st.spinner("Testing new API key..."):
                                    is_valid, message = test_api_key(new_api_key)
                                    if is_valid:
                                        st.session_state.temp_api_key = new_api_key
                                        if 'agent' in st.session_state:
                                            del st.session_state.agent
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                    
                    st.markdown("""
                    **Common solutions:**
                    - Get a fresh API key from [OpenRouter.ai](https://openrouter.ai/)
                    - Check that your account is in good standing
                    - Verify you have credits/usage remaining
                    """)
                return
    
    agent = st.session_state.agent
    
    # Sidebar for file management and quick info
    with st.sidebar:
        # Theme Toggle at the top
        st.markdown("### 🎨 Theme")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("☀️ Light", 
                        type="primary" if st.session_state.theme_mode == 'light' else "secondary",
                        use_container_width=True,
                        key="light_theme"):
                st.session_state.theme_mode = 'light'
                st.rerun()
        
        with col2:
            if st.button("🌙 Dark", 
                        type="primary" if st.session_state.theme_mode == 'dark' else "secondary",
                        use_container_width=True,
                        key="dark_theme"):
                st.session_state.theme_mode = 'dark'
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Control Panel")
        
        # Quick stats if files are processed
        if agent.document_content:
            st.markdown("### � Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Files", len(agent.document_content))
            with col2:
                st.metric("📊 Datasets", len(agent.data_frames))
        
        # API Key Management
        with st.expander("🔑 API Configuration"):
            current_key = os.getenv('OPENROUTER_API_KEY')
            if current_key:
                st.success("✅ API key loaded")
                st.text(f"Key: {current_key[:8]}...")
            else:
                st.warning("⚠️ No API key found")
        
        # File upload section (now moved to main tab)
        st.markdown("### 🧭 Quick Navigation")
        st.markdown("""
        - **🏠 Home**: Overview & features
        - **📤 Upload**: Upload documents here ⬅️
        - **💬 Chat**: Ask questions about your files
        - **📊 Analytics**: View data insights
        - **⚙️ Settings**: Configure app settings
        """)
        
        # Processed files display
        if agent.document_content:
            st.markdown("### 📋 Processed Files")
            for file_name in agent.document_content.keys():
                file_type = agent.document_content[file_name]['file_type']
                icon = "📊" if file_type in ['csv', 'xlsx', 'xls'] else "📄"
                st.text(f"{icon} {file_name}")
        
        # Help section
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **🚀 Getting Started:**
            1. Go to **📤 Upload & Process** tab
            2. Upload your documents there
            3. Wait for AI processing
            4. Chat with your documents!
            
            **📧 Support:** support@archusers.com
            """)
        
        # Clear button
        if agent.document_content:
            if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
                agent.document_content.clear()
                agent.data_frames.clear()
                agent.analysis_results.clear()
                agent.conversation_history.clear()
                st.success("✨ All files cleared!")
                st.rerun()
        
        # App info
        st.markdown("---")
        st.markdown("**🤖 AI Document Analyst v2.0**")
        st.markdown("Built by ARCH_USERS")
        st.markdown("Powered by Together AI")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home", "📤 Upload & Process", "💬 AI Chat", "📊 Analytics", 
        "⚙️ Settings", "⚖️ Legal Analysis", "🔒 Legal Security"
    ])
    
    with tab1:
        # Welcome and features section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🎯 What can I do for you?")
            
            features = [
                ("� Multi-Format Support", "PDF, DOCX, TXT, CSV, Excel, Images - I handle them all!"),
                ("🧠 AI-Powered Analysis", "Smart summaries and insights using advanced AI models"),
                ("📊 Data Visualization", "Automatic charts, graphs, and statistical analysis"),
                ("💬 Conversational Q&A", "Ask questions in natural language, get intelligent answers"),
                ("📈 Comprehensive Reports", "Executive summaries with key findings and recommendations"),
                ("⚡ Real-time Processing", "Fast document processing with live progress tracking")
            ]
            
            for title, desc in features:
                st.markdown(f"""
                <div class="feature-card">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. **📤 Upload** your documents in the **Upload & Process** tab
            2. **🔄 Process** files to extract insights
            3. **💬 Chat** with your documents using AI
            4. **📊 Analyze** data with automatic visualizations
            5. **📈 Export** reports and findings
            """)
            
            if not agent.document_content:
                st.info("� Go to **Upload & Process** tab to start!")
                st.markdown("**🎯 Next Step:** Click the **📤 Upload & Process** tab above")
            else:
                st.success(f"🎉 {len(agent.document_content)} files ready for analysis!")
    
    with tab2:
        # Upload and processing interface
        st.markdown("## 📤 Document Upload & Processing")
        
        # File upload section in the main tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📁 Upload Your Documents")
            uploaded_files = st.file_uploader(
                "Choose files to analyze",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'jpg', 'jpeg', 'png', 'xls'],
                help="Supported formats: PDF, DOCX, TXT, CSV, Excel, Images (JPG, PNG)",
                key="main_uploader"
            )
            
            # Upload instructions
            st.markdown("""
            **📋 Instructions:**
            - **Drag & drop** files or **click** to browse
            - **Multiple files** can be uploaded at once
            - **Supported formats**: PDF, Word, Text, CSV, Excel, Images
            - **File size limit**: 200MB per file
            """)
        
        with col2:
            st.markdown("### 📊 Upload Stats")
            if agent.document_content:
                st.metric("📄 Total Files", len(agent.document_content))
                st.metric("📊 Data Files", len(agent.data_frames))
                st.metric("💬 Conversations", len(agent.conversation_history))
            else:
                st.info("No files uploaded yet")
            
            # Quick actions
            if agent.document_content:
                if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
                    agent.document_content.clear()
                    agent.data_frames.clear()
                    agent.analysis_results.clear()
                    agent.conversation_history.clear()
                    st.success("✨ All files cleared!")
                    st.rerun()
        
        if uploaded_files:
            st.markdown("### 🔄 Processing Documents...")
            
            # Create a progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Check if already processed
                    if uploaded_file.name in agent.document_content:
                        status_text.text(f"✅ {uploaded_file.name} already processed")
                        continue
                    
                    status_text.text(f"🔄 Processing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Process file
                        result = agent.process_document(temp_path, uploaded_file.name)
                        
                        # Display processing result
                        with st.expander(f"✅ {uploaded_file.name} - Processed Successfully", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("**📝 AI Summary:**")
                                st.write(result['summary'])
                            
                            with col2:
                                st.markdown("**� File Info:**")
                                st.text(f"Type: {result['file_type'].upper()}")
                                st.text(f"Size: {len(result['content'])} chars")
                                
                                if result['data_frame'] is not None:
                                    df = result['data_frame']
                                    st.text(f"Rows: {df.shape[0]}")
                                    st.text(f"Columns: {df.shape[1]}")
                        
                        # If structured data, show preview and analysis
                        if uploaded_file.name in agent.data_frames:
                            df = agent.data_frames[uploaded_file.name]
                            
                            with st.expander(f"📊 Data Preview - {uploaded_file.name}"):
                                # Data preview
                                st.markdown("**First 10 rows:**")
                                st.dataframe(df.head(10), use_container_width=True)
                                
                                # Quick stats
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📏 Rows", df.shape[0])
                                with col2:
                                    st.metric("📋 Columns", df.shape[1])
                                with col3:
                                    st.metric("🔢 Numeric", len(df.select_dtypes(include=['number']).columns))
                                with col4:
                                    st.metric("📝 Text", len(df.select_dtypes(include=['object']).columns))
                            
                            # Auto-generate visualizations
                            with st.spinner("🎨 Creating visualizations..."):
                                viz_paths = agent.create_visualizations(df, uploaded_file.name)
                                
                                if viz_paths:
                                    with st.expander(f"📈 Auto-Generated Charts - {uploaded_file.name}"):
                                        # Display visualizations in columns
                                        viz_cols = st.columns(2)
                                        for idx, viz_path in enumerate(viz_paths):
                                            if os.path.exists(viz_path):
                                                with viz_cols[idx % 2]:
                                                    chart_name = os.path.basename(viz_path).replace('.png', '').replace('_', ' ').title()
                                                    st.markdown(f"**{chart_name}**")
                                                    st.image(viz_path, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ All files processed successfully!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
        
        else:
            # Empty state with helpful instructions
            st.markdown("""
            <div class="upload-zone">
                <h3>📁 Drop your documents here!</h3>
                <p>Supported formats: PDF, DOCX, TXT, CSV, Excel, Images</p>
                <p>Use the file uploader above to get started ⬆️</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 What happens when you upload?")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔍 Text Extraction**
                - PDF text extraction
                - Word document parsing
                - Image OCR processing
                """)
            
            with col2:
                st.markdown("""
                **🧠 AI Analysis**
                - Intelligent summaries
                - Key insights extraction
                - Pattern recognition
                """)
            
            with col3:
                st.markdown("""
                **📊 Data Processing**
                - Automatic statistics
                - Chart generation
                - Correlation analysis
                """)

    with tab3:
        st.markdown("## 💬 Chat with Your Documents")
        
        if not agent.document_content:
            st.info("� Upload documents first to start chatting!")
            st.markdown("""
            ### 🎯 What you can ask:
            - "What are the key insights from this data?"
            - "Summarize the main points of this document"
            - "What patterns do you see in the numbers?"
            - "What are the most important findings?"
            - "Show me correlations between variables"
            """)
        else:
            # Chat interface
            st.markdown("### 🗨️ Ask anything about your documents")
            
            # Chat input
            user_question = st.text_area(
                "Your Question:",
                placeholder="Ask anything about your uploaded documents...",
                height=100,
                help="Type your question and press Ctrl+Enter or click the button below"
            )
            
            # Enhanced ask button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                ask_button = st.button("🔍 Get AI Answer", type="primary", use_container_width=True)
            
            if ask_button and user_question.strip():
                with st.spinner("� AI is thinking..."):
                    try:
                        answer = agent.answer_question(user_question)
                        
                        # Display answer in a nice format
                        st.markdown("### 💡 AI Response:")
                        st.markdown(f"""
                        <div class="chat-container">
                            <p><strong>❓ Your Question:</strong> {user_question}</p>
                            <p><strong>🤖 AI Answer:</strong> {answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")
            
            elif ask_button:
                st.warning("Please enter a question first!")
            
            # Quick question buttons for data files
            if agent.data_frames:
                st.markdown("### 🚀 Quick Questions")
                st.markdown("*Click any button for instant insights:*")
                
                quick_questions = [
                    ("📊 Key Statistics", "What are the key statistics and summary of this dataset?"),
                    ("📈 Trends & Patterns", "What trends and patterns do you see in this data?"),
                    ("🔍 Data Quality", "Are there any missing values or data quality issues?"),
                    ("💡 Key Insights", "What are the most important insights from this data?"),
                    ("🔗 Correlations", "What correlations exist between different variables?"),
                    ("📋 Executive Summary", "Provide an executive summary of the findings")
                ]
                
                # Display buttons in a grid
                for i in range(0, len(quick_questions), 2):
                    col1, col2 = st.columns(2)
                    
                    # First button
                    with col1:
                        if i < len(quick_questions):
                            title, question = quick_questions[i]
                            if st.button(title, key=f"quick_{i}", use_container_width=True):
                                with st.spinner("🤖 Analyzing..."):
                                    try:
                                        answer = agent.answer_question(question)
                                        st.success("💡 **Answer:**")
                                        st.write(answer)
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                    
                    # Second button
                    with col2:
                        if i + 1 < len(quick_questions):
                            title, question = quick_questions[i + 1]
                            if st.button(title, key=f"quick_{i+1}", use_container_width=True):
                                with st.spinner("� Analyzing..."):
                                    try:
                                        answer = agent.answer_question(question)
                                        st.success("💡 **Answer:**")
                                        st.write(answer)
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
            
            # Conversation history
            if agent.conversation_history:
                st.markdown("### 💭 Recent Conversations")
                
                # Show last 3 conversations
                for i, item in enumerate(reversed(agent.conversation_history[-3:])):
                    with st.expander(f"💬 Q{len(agent.conversation_history)-i}: {item['question'][:60]}..."):
                        st.markdown(f"**❓ Question:** {item['question']}")
                        st.markdown(f"**🤖 Answer:** {item['answer']}")

    with tab4:
        # Analytics Dashboard
        st.markdown("## 📊 Analytics Dashboard")
        
        if not agent.data_frames:
            st.info("📈 Upload CSV or Excel files to see analytics!")
            st.markdown("""
            ### 📊 Available Analytics:
            - **Statistical Summary**: Mean, median, mode, standard deviation
            - **Data Quality Check**: Missing values, duplicates, outliers
            - **Correlation Analysis**: Relationships between variables
            - **Distribution Plots**: Histograms, box plots, scatter plots
            - **Trend Analysis**: Time series and pattern recognition
            """)
        else:
            # Display analytics for each dataset
            for file_name, df in agent.data_frames.items():
                with st.expander(f"📊 Analytics: {file_name}", expanded=True):
                    # Basic metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📏 Rows", df.shape[0])
                    with col2:
                        st.metric("📋 Columns", df.shape[1])
                    with col3:
                        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                        st.metric("❌ Missing %", f"{missing_percent:.1f}%")
                    with col4:
                        numeric_cols = len(df.select_dtypes(include=['number']).columns)
                        st.metric("🔢 Numeric", numeric_cols)
                    
                    # Statistical summary for numeric columns
                    numeric_df = df.select_dtypes(include=['number'])
                    if not numeric_df.empty:
                        st.markdown("**📈 Statistical Summary:**")
                        st.dataframe(numeric_df.describe(), use_container_width=True)
                    
                    # Show visualizations if they exist
                    viz_dir = f"visualizations_{file_name.replace('.', '_')}"
                    if os.path.exists(viz_dir):
                        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
                        if viz_files:
                            st.markdown("**📊 Visualizations:**")
                            for viz_file in viz_files:
                                viz_path = os.path.join(viz_dir, viz_file)
                                chart_name = viz_file.replace('.png', '').replace('_', ' ').title()
                                st.markdown(f"*{chart_name}*")
                                st.image(viz_path, use_container_width=True)

    with tab5:
        # Settings Tab
        st.markdown("## ⚙️ Application Settings")
        
        # API Configuration Section
        st.markdown("### 🔑 API Key Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Current API Key Status
            current_key = os.getenv('OPENROUTER_API_KEY')
            if current_key:
                st.success(f"✅ **Current API Key Status:** Active")
                st.text(f"🔐 Key Preview: {current_key[:12]}...{current_key[-8:]}")
                st.text(f"📅 Loaded from: Environment (.env file)")
            else:
                st.error("❌ **No API Key Found**")
                st.warning("Please set your OPENROUTER_API_KEY in the .env file or use temporary override below.")
            
            # Temporary API Key Override
            st.markdown("#### 🔄 Temporary API Key Override")
            st.info("💡 This will override your .env API key for this session only")
            
            temp_api_key = st.text_input(
                "Enter Temporary API Key:",
                type="password",
                placeholder="Enter your OpenRouter API key here...",
                help="This will be used instead of the .env file key until you refresh the page"
            )
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_a:
                if st.button("🔄 Apply Temporary Key", type="primary", use_container_width=True):
                    if temp_api_key.strip():
                        try:
                            # Test the API key by creating a new agent
                            test_agent = DocumentAnalystAgent(temp_api_key)
                            st.session_state.agent = test_agent
                            st.session_state.temp_api_key = temp_api_key
                            st.success("✅ Temporary API key applied successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to apply API key: {str(e)}")
                    else:
                        st.warning("Please enter a valid API key")
            
            with col_b:
                if st.button("🔄 Reset to .env Key", type="secondary", use_container_width=True):
                    if current_key:
                        try:
                            st.session_state.agent = DocumentAnalystAgent(current_key)
                            if 'temp_api_key' in st.session_state:
                                del st.session_state.temp_api_key
                            st.success("✅ Reset to .env API key!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to reset: {str(e)}")
                    else:
                        st.error("No .env API key found to reset to")
            
            with col_c:
                if st.button("🧪 Test Current Key", use_container_width=True):
                    try:
                        # Test the current agent's API key with a simple call
                        test_response = agent._make_api_call_with_retry("Hello, this is a test.", max_tokens=10)
                        if "error" not in test_response.lower():
                            st.success("✅ API key is working correctly!")
                        else:
                            st.error(f"❌ API key test failed: {test_response}")
                    except Exception as e:
                        st.error(f"❌ API key test failed: {str(e)}")
        
        with col2:
            st.markdown("#### 📖 How to get API Key")
            st.markdown("""
            1. Visit [OpenRouter](https://openrouter.ai/)
            2. Sign up or log in to your account
            3. Navigate to API Keys section
            4. Create a new API key
            5. Copy and paste it here or in your .env file
            """)
            
            if st.button("🌐 Open OpenRouter", use_container_width=True):
                st.markdown("[🔗 Click here to visit OpenRouter](https://openrouter.ai/)")
        
        st.markdown("---")
        
        # Model Configuration Section
        st.markdown("### 🤖 AI Model Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Current Model:** `{agent.model}`")
            
            # Available models with descriptions
            available_models = {
                "meta-llama/Llama-3.1-8B-Instruct-Turbo": {
                    "name": "Llama 3.1 8B Turbo (Recommended)",
                    "description": "Fast, efficient, good for most tasks",
                    "rate_limit": "Standard",
                    "performance": "⭐⭐⭐⭐"
                },
                "meta-llama/Llama-3.1-70B-Instruct-Turbo": {
                    "name": "Llama 3.1 70B Turbo",
                    "description": "More powerful, better reasoning",
                    "rate_limit": "Standard",
                    "performance": "⭐⭐⭐⭐⭐"
                },
                "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {
                    "name": "Llama 3.2 11B Vision",
                    "description": "Vision-capable model",
                    "rate_limit": "Standard",
                    "performance": "⭐⭐⭐⭐"
                },
                "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                    "name": "Mixtral 8x7B",
                    "description": "Alternative high-performance model",
                    "rate_limit": "Standard",
                    "performance": "⭐⭐⭐⭐"
                },
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
                    "name": "Nous Hermes 2 Mixtral",
                    "description": "Optimized for conversations",
                    "rate_limit": "Standard",
                    "performance": "⭐⭐⭐⭐"
                }
            }
            
            # Model selection
            model_names = list(available_models.keys())
            current_index = model_names.index(agent.model) if agent.model in model_names else 0
            
            selected_model = st.selectbox(
                "Select AI Model:",
                options=model_names,
                format_func=lambda x: available_models[x]["name"],
                index=current_index,
                help="Choose the AI model that best fits your needs"
            )
            
            # Show model details
            if selected_model:
                model_info = available_models[selected_model]
                st.markdown(f"""
                **Model Details:**
                - **Description:** {model_info['description']}
                - **Rate Limit:** {model_info['rate_limit']}
                - **Performance:** {model_info['performance']}
                """)
            
            # Apply model change
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("🔄 Apply Model Change", type="primary", use_container_width=True):
                    if selected_model != agent.model:
                        agent.model = selected_model
                        st.success(f"✅ Model changed to: {available_models[selected_model]['name']}")
                        st.rerun()
                    else:
                        st.info("Model is already selected")
            
            with col_b:
                if st.button("🧪 Test Selected Model", use_container_width=True):
                    try:
                        # Temporarily test the selected model
                        old_model = agent.model
                        agent.model = selected_model
                        test_response = agent._make_api_call_with_retry("Respond with 'Model test successful'", max_tokens=10)
                        agent.model = old_model  # Restore original model
                        
                        if "successful" in test_response.lower():
                            st.success("✅ Model test successful!")
                        else:
                            st.warning(f"⚠️ Model responded: {test_response}")
                    except Exception as e:
                        st.error(f"❌ Model test failed: {str(e)}")
        
        with col2:
            st.markdown("#### 📊 Model Comparison")
            st.markdown("""
            **🚀 Turbo Models:**
            - Faster response times
            - Lower latency
            - Good for real-time applications
            
            **🧠 Large Models (70B):**
            - Better reasoning
            - More accurate responses
            - Higher quality analysis
            
            **👁️ Vision Models:**
            - Can process images
            - Multimodal capabilities
            - Text + image understanding
            """)
        
        st.markdown("---")
        
        # Theme Configuration Section
        st.markdown("### 🎨 Theme & Appearance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Current Theme:** {st.session_state.theme_mode.title()} Mode")
            
            # Theme selection
            theme_options = {
                'light': {
                    'name': '☀️ Light Mode',
                    'description': 'Clean, bright interface with white backgrounds',
                    'preview': '🤍 White backgrounds, dark text'
                },
                'dark': {
                    'name': '🌙 Dark Mode',
                    'description': 'Modern dark interface, easier on the eyes',
                    'preview': '🖤 Dark backgrounds, light text'
                }
            }
            
            selected_theme = st.selectbox(
                "Choose Theme:",
                options=list(theme_options.keys()),
                format_func=lambda x: theme_options[x]["name"],
                index=1 if st.session_state.theme_mode == 'dark' else 0,
                help="Select your preferred visual theme"
            )
            
            # Show theme details
            if selected_theme:
                theme_info = theme_options[selected_theme]
                st.markdown(f"""
                **Theme Details:**
                - **Description:** {theme_info['description']}
                - **Preview:** {theme_info['preview']}
                """)
            
            # Apply theme change
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("🎨 Apply Theme", type="primary", use_container_width=True):
                    if selected_theme != st.session_state.theme_mode:
                        st.session_state.theme_mode = selected_theme
                        st.success(f"✅ Theme changed to: {theme_options[selected_theme]['name']}")
                        st.rerun()
                    else:
                        st.info("Theme is already selected")
            
            with col_b:
                if st.button("🔄 Reset to Default", use_container_width=True):
                    st.session_state.theme_mode = 'dark'
                    st.success("✅ Theme reset to Dark Mode!")
                    st.rerun()
        
        with col2:
            st.markdown("#### 🎨 Theme Preview")
            
            # Theme preview cards
            if st.session_state.theme_mode == 'dark':
                st.markdown("""
                <div style="background: #2d2d2d; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>🌙 Dark Mode Active</strong><br>
                    • Reduced eye strain<br>
                    • Better for low light<br>
                    • Modern appearance
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f8f9fa; color: black; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #dee2e6;">
                    <strong>☀️ Light Mode Active</strong><br>
                    • Classic clean look<br>
                    • High contrast text<br>
                    • Professional appearance
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Theme Tips")
            st.markdown("""
            **🌙 Dark Mode Benefits:**
            - Reduces eye strain in low light
            - Saves battery on OLED screens
            - Modern, sleek appearance
            
            **☀️ Light Mode Benefits:**
            - Better readability in bright environments
            - Classic, professional look
            - Higher contrast for text
            """)
        
        st.markdown("---")
        
        # Application Settings
        st.markdown("### 🎛️ Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Processing Settings")
            
            # Max tokens setting
            max_tokens = st.slider(
                "Max Response Tokens:",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                help="Maximum number of tokens for AI responses"
            )
            
            # Temperature setting
            temperature = st.slider(
                "AI Creativity (Temperature):",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values make responses more creative but less focused"
            )
            
            # Retry attempts
            max_retries = st.slider(
                "Max Retry Attempts:",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of times to retry failed API calls"
            )
            
            if st.button("💾 Save Processing Settings", use_container_width=True):
                st.session_state.max_tokens = max_tokens
                st.session_state.temperature = temperature
                st.session_state.max_retries = max_retries
                st.success("✅ Processing settings saved!")
        
        with col2:
            st.markdown("#### 📈 Session Information")
            
            # Session stats
            st.metric("📄 Processed Files", len(agent.document_content))
            st.metric("📊 Datasets Loaded", len(agent.data_frames))
            st.metric("💬 Conversations", len(agent.conversation_history))
            
            # Current settings display
            st.markdown("**Current Settings:**")
            st.text(f"Max Tokens: {getattr(st.session_state, 'max_tokens', 500)}")
            st.text(f"Temperature: {getattr(st.session_state, 'temperature', 0.3)}")
            st.text(f"Max Retries: {getattr(st.session_state, 'max_retries', 3)}")
            
            # Session management
            if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
                # Clear all session data
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session reset! Please refresh the page.")
        
        st.markdown("---")
        
        # About Section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **📊 AI Document Analyst v2.0**
        - Built by ARCH_USERS
        - Powered by Together AI & Meta Llama
        - GitHub: [Dataa_Analyst_Agent](https://github.com/ARCH_USERS/Dataa_Analyst_Agent)
        - Email: support@archusers.com
        
        **🚀 Features:**
        - Multi-format document processing
        - AI-powered analysis and insights
        - Interactive chat interface
        - Automatic data visualization
        - Comprehensive analytics dashboard
        - Legal document analysis and insights
        - Risk assessment and compliance checking
        - Legal security and privacy protection
        """)

    with tab6:
        # Legal Analysis Tab
        st.markdown("## ⚖️ Legal Document Analysis")
        
        if not agent.document_content:
            st.info("📄 Upload legal documents first to perform legal analysis!")
            st.markdown("""
            ### 🎯 Legal Analysis Features:
            - **Document Classification**: Automatically identify contract types, statutes, regulations
            - **Risk Assessment**: Detect and evaluate legal risks and liabilities
            - **Plain English Translation**: Convert legal jargon to understandable language
            - **Citation Analysis**: Extract and validate legal citations
            - **Compliance Checking**: Verify document completeness and compliance
            - **Entity Recognition**: Identify parties, courts, jurisdictions
            - **Date Extraction**: Find critical deadlines and dates
            - **Document Comparison**: Compare contract versions (redlining)
            """)
        else:
            # Legal document selection
            legal_docs = [name for name, doc in agent.document_content.items() 
                         if agent.is_legal_document(doc['content'])]
            
            if not legal_docs:
                st.warning("⚠️ No legal documents detected in uploaded files")
                st.markdown("""
                **Legal Document Indicators:**
                - Contracts and agreements
                - Legal briefs and motions
                - Statutes and regulations
                - Court decisions
                - Patents and IP documents
                """)
            else:
                st.success(f"✅ {len(legal_docs)} legal document(s) detected")
                
                selected_doc = st.selectbox(
                    "Select Document for Legal Analysis:",
                    legal_docs,
                    help="Choose a legal document to analyze"
                )
                
                if selected_doc:
                    # Display basic legal analysis
                    if selected_doc in agent.legal_analysis_results:
                        analysis = agent.legal_analysis_results[selected_doc]
                        
                        # Legal Analysis Overview
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            doc_type = analysis.get('document_type', 'Unknown')
                            st.metric("📋 Document Type", doc_type.replace('_', ' ').title())
                        
                        with col2:
                            risk_level = analysis.get('risk_assessment', {}).get('overall_risk', 'Unknown')
                            risk_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(risk_level, '⚪')
                            st.metric("⚠️ Risk Level", f"{risk_color} {risk_level.title()}")
                        
                        with col3:
                            compliance_score = analysis.get('compliance_check', {}).get('score', 0)
                            st.metric("✅ Compliance Score", f"{compliance_score}%")
                        
                        # Detailed Analysis Sections
                        st.markdown("### 📊 Detailed Legal Analysis")
                        
                        # Risk Assessment
                        with st.expander("⚠️ Risk Assessment", expanded=True):
                            risk_assessment = analysis.get('risk_assessment', {})
                            
                            if risk_assessment.get('risk_factors'):
                                st.markdown("**Identified Risk Factors:**")
                                for risk in risk_assessment['risk_factors']:
                                    level_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(risk['level'], '⚪')
                                    st.markdown(f"- {level_icon} **{risk['factor'].title()}** ({risk['level']} risk)")
                                    if risk.get('context'):
                                        st.text(f"   Context: {risk['context'][:100]}...")
                            
                            if risk_assessment.get('recommendations'):
                                st.markdown("**Recommendations:**")
                                for rec in risk_assessment['recommendations']:
                                    st.markdown(f"- 💡 {rec}")
                        
                        # Plain English Summary
                        with st.expander("📝 Plain English Summary", expanded=True):
                            plain_summary = analysis.get('plain_english_summary', '')
                            if plain_summary:
                                st.write(plain_summary)
                            else:
                                if st.button("🔄 Generate Plain English Summary"):
                                    with st.spinner("Translating legal text..."):
                                        summary = agent.legal_analyzer.translate_to_plain_english(
                                            agent.document_content[selected_doc]['content']
                                        )
                                        st.write(summary)
                        
                        # Legal Entities
                        with st.expander("👥 Legal Entities"):
                            entities = analysis.get('entities', [])
                            if entities:
                                for entity in entities[:10]:
                                    st.markdown(f"- **{entity.name}** ({entity.entity_type}) - Role: {entity.role}")
                            else:
                                st.info("No legal entities extracted")
                        
                        # Legal Citations
                        with st.expander("📚 Legal Citations"):
                            citations = analysis.get('citations', [])
                            if citations:
                                for citation in citations[:10]:
                                    st.markdown(f"- **{citation.text}** ({citation.citation_type})")
                                    if citation.case_name:
                                        st.text(f"  Case: {citation.case_name}")
                                    if citation.court:
                                        st.text(f"  Court: {citation.court}")
                            else:
                                st.info("No legal citations found")
                        
                        # Key Obligations
                        with st.expander("📋 Key Obligations"):
                            obligations = analysis.get('key_obligations', [])
                            if obligations:
                                for i, obligation in enumerate(obligations[:10], 1):
                                    st.markdown(f"{i}. {obligation}")
                            else:
                                st.info("No specific obligations identified")
                        
                        # Legal Dates and Deadlines
                        with st.expander("📅 Important Dates"):
                            dates = analysis.get('dates', [])
                            if dates:
                                for date in dates[:10]:
                                    importance_icon = {'critical': '🔴', 'important': '🟡', 'informational': '🔵'}.get(date.importance, '⚪')
                                    st.markdown(f"- {importance_icon} **{date.date_text}** ({date.date_type})")
                            else:
                                st.info("No specific dates extracted")
                        
                        # Actions
                        st.markdown("### 🎯 Actions")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("📊 Generate Legal Visualizations", use_container_width=True):
                                with st.spinner("Creating legal visualizations..."):
                                    viz_paths = agent.create_legal_visualizations(selected_doc)
                                    if viz_paths:
                                        st.success(f"✅ Generated {len(viz_paths)} visualizations")
                                        for viz_path in viz_paths:
                                            st.image(viz_path, use_container_width=True)
                                    else:
                                        st.warning("No visualizations could be generated")
                        
                        with col2:
                            if st.button("📄 Generate Legal Report", use_container_width=True):
                                with st.spinner("Generating comprehensive legal report..."):
                                    report = agent.generate_legal_report(selected_doc)
                                    st.markdown("### 📋 Legal Analysis Report")
                                    st.write(report)
                        
                        with col3:
                            if st.button("🔄 Refresh Analysis", use_container_width=True):
                                with st.spinner("Re-analyzing document..."):
                                    content = agent.document_content[selected_doc]['content']
                                    new_analysis = agent.legal_analyzer.analyze_legal_document(content)
                                    agent.legal_analysis_results[selected_doc] = new_analysis
                                    st.success("✅ Analysis refreshed")
                                    st.rerun()
                    
                    else:
                        st.warning("⚠️ Legal analysis not available. Processing document...")
                        with st.spinner("Performing legal analysis..."):
                            analysis = agent.perform_legal_analysis(selected_doc)
                            if 'error' not in analysis:
                                agent.legal_analysis_results[selected_doc] = analysis
                                st.success("✅ Legal analysis completed")
                                st.rerun()
                            else:
                                st.error(f"❌ Analysis failed: {analysis['error']}")
        
        # Document Comparison Section
        st.markdown("---")
        st.markdown("### 📊 Legal Document Comparison")
        
        legal_docs = [name for name, doc in agent.document_content.items() 
                     if agent.is_legal_document(doc['content'])]
        
        if len(legal_docs) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                doc1 = st.selectbox("First Document:", legal_docs, key="legal_comp_doc1")
            
            with col2:
                doc2 = st.selectbox("Second Document:", legal_docs, key="legal_comp_doc2")
            
            if doc1 and doc2 and doc1 != doc2:
                if st.button("🔍 Compare Documents", use_container_width=True):
                    with st.spinner("Comparing legal documents..."):
                        comparison = agent.compare_legal_documents(doc1, doc2)
                        
                        if 'error' not in comparison:
                            st.markdown("#### 📋 Comparison Results")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("📊 Similarity Score", f"{comparison['similarity_score']}%")
                            with col2:
                                status = "✅ Identical" if comparison['similarity_score'] > 95 else "📊 Different"
                                st.metric("📄 Status", status)
                            
                            if comparison.get('added_content'):
                                with st.expander("➕ Added Content"):
                                    for line in comparison['added_content'][:10]:
                                        st.text(f"+ {line.strip()}")
                            
                            if comparison.get('removed_content'):
                                with st.expander("➖ Removed Content"):
                                    for line in comparison['removed_content'][:10]:
                                        st.text(f"- {line.strip()}")
                        else:
                            st.error(f"❌ Comparison failed: {comparison['error']}")
        else:
            st.info("📄 Upload at least 2 legal documents to enable comparison")

    with tab7:
        # Legal Security Tab
        st.markdown("## 🔒 Legal Security & Privacy")
        
        if not agent.document_content:
            st.info("📄 Upload documents first to perform security analysis!")
            st.markdown("""
            ### 🛡️ Legal Security Features:
            - **Privilege Detection**: Identify attorney-client privileged content
            - **Confidentiality Checking**: Detect confidential information markers
            - **Access Logging**: Track document access and usage
            - **Data Encryption**: Secure sensitive legal information
            - **Audit Trail**: Maintain comprehensive access logs
            - **Compliance Notices**: Generate confidentiality warnings
            """)
        else:
            # Privilege and Confidentiality Check
            st.markdown("### 🛡️ Privilege & Confidentiality Analysis")
            
            privileged_docs = []
            
            for doc_name, doc_info in agent.document_content.items():
                privilege_check = agent.security_manager.check_privilege(doc_info['content'])
                if privilege_check['is_privileged']:
                    privileged_docs.append((doc_name, privilege_check))
            
            if privileged_docs:
                st.warning(f"⚠️ {len(privileged_docs)} document(s) contain privileged content")
                
                for doc_name, privilege_info in privileged_docs:
                    with st.expander(f"🔒 {doc_name} - Privileged Content Detected"):
                        st.error(privilege_info['warning'])
                        st.markdown("**Privilege Markers Found:**")
                        for marker in privilege_info['privilege_markers']:
                            st.markdown(f"- 🔸 {marker}")
                        
                        # Show confidentiality notice
                        st.markdown("**Confidentiality Notice:**")
                        st.text(agent.security_manager.generate_confidentiality_notice())
            else:
                st.success("✅ No privileged content detected in uploaded documents")
            
            # Access Logging
            st.markdown("### 📋 Access Audit Trail")
            
            if agent.security_manager.access_logs:
                st.success(f"📊 {len(agent.security_manager.access_logs)} access log entries")
                
                # Display recent access logs
                with st.expander("📜 Recent Access Logs"):
                    for log in agent.security_manager.access_logs[-10:]:
                        st.markdown(f"**{log['timestamp']}** - User: {log['user_id']} - Action: {log['action']} - Document: {log['document']}")
            else:
                st.info("📝 No access logs recorded yet")
            
            # Document Security Actions
            st.markdown("### 🎯 Security Actions")
            
            selected_doc = st.selectbox(
                "Select Document for Security Analysis:",
                list(agent.document_content.keys()),
                help="Choose a document to analyze for security issues"
            )
            
            if selected_doc:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔍 Check Privilege", use_container_width=True):
                        content = agent.document_content[selected_doc]['content']
                        privilege_check = agent.security_manager.check_privilege(content)
                        
                        if privilege_check['is_privileged']:
                            st.error("🔒 PRIVILEGED CONTENT DETECTED")
                            st.warning(privilege_check['warning'])
                            for marker in privilege_check['privilege_markers']:
                                st.text(f"• {marker}")
                        else:
                            st.success("✅ No privileged content markers found")
                
                with col2:
                    if st.button("📝 Log Access", use_container_width=True):
                        agent.security_manager.log_access(
                            user_id="current_user",
                            document_name=selected_doc,
                            action="security_review"
                        )
                        st.success(f"✅ Access logged for {selected_doc}")
                
                with col3:
                    if st.button("🔐 Generate Notice", use_container_width=True):
                        notice = agent.security_manager.generate_confidentiality_notice()
                        st.markdown("### 📋 Confidentiality Notice")
                        st.text(notice)
            
            # Security Settings
            st.markdown("---")
            st.markdown("### ⚙️ Security Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔐 Encryption Settings")
                
                enable_encryption = st.checkbox(
                    "Enable Document Encryption",
                    help="Encrypt sensitive document content"
                )
                
                auto_privilege_check = st.checkbox(
                    "Automatic Privilege Detection",
                    value=True,
                    help="Automatically check for privileged content on upload"
                )
                
                log_all_access = st.checkbox(
                    "Log All Document Access",
                    value=True,
                    help="Maintain audit trail of all document interactions"
                )
            
            with col2:
                st.markdown("#### 📊 Security Statistics")
                
                total_docs = len(agent.document_content)
                privileged_count = len(privileged_docs)
                access_logs_count = len(agent.security_manager.access_logs)
                
                st.metric("📄 Total Documents", total_docs)
                st.metric("🔒 Privileged Documents", privileged_count)
                st.metric("📝 Access Log Entries", access_logs_count)
                
                # Security score
                if total_docs > 0:
                    security_score = max(0, 100 - (privileged_count / total_docs * 50))
                    st.metric("🛡️ Security Score", f"{security_score:.0f}%")


def smart_streamlit_launch():
    """Smart Streamlit launcher"""
    port = 8502
    url = f"http://localhost:{port}"
    
    try:
        print("🚀 Starting Document Analyst Agent...")
        print(f"📊 Streamlit will be available at {url}")
        print("🌐 Please open the URL manually in your browser")
        
        script_path = os.path.abspath(__file__)
        
        # Launch Streamlit without auto-opening browser
        cmd = [
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ]
        
        print("🛑 Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Detect if running in Streamlit
    try:
        # Check if we're running within Streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            create_streamlit_ui()
        else:
            smart_streamlit_launch()
    except ImportError:
        # Fallback for older Streamlit versions
        try:
            import streamlit as st
            if hasattr(st, '_is_running_with_streamlit'):
                create_streamlit_ui()
            else:
                smart_streamlit_launch()
        except:
            smart_streamlit_launch()