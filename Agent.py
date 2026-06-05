'''
AI Document Analyst - core agent.

This module exposes DocumentAnalystAgent, a reusable engine that:
    - extracts text/data from PDF, DOCX, TXT, CSV, Excel, and images (OCR)
    - performs pandas-based data analysis and matplotlib/seaborn visualisations
    - calls an OpenAI-compatible chat completions API for summaries and Q&A

The default backend is OpenCode Zen (https://opencode.ai/zen). The endpoint,
auth header, and request shape follow the OpenAI chat-completions spec, so
the client is just a thin `requests` wrapper - no SDK lock-in.

Environment variables (in priority order):
    1. st.secrets["OPENCODE_ZEN_API_KEY"]   - Streamlit Cloud secrets
    2. OPENCODE_ZEN_API_KEY env var        - .env file / system env
    3. TOGETHER_API_KEY env var            - legacy, for back-compat

If for privacy reasons you don't trust the app, use your own key.
'''

import os
import sys
import warnings
import time
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Mark this module as running under Streamlit so the noisy status prints
# below stay quiet in the deployed app.
os.environ.setdefault("STREAMLIT_RUN", "0")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    # quiet success log when imported under Streamlit (avoids console spam)
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("✅ Environment variables loaded")
except ImportError:
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("⚠️  python-dotenv not found. Please install: pip install python-dotenv")
except Exception as e:
    if os.environ.get("STREAMLIT_RUN") != "1":
        print(f"⚠️  Error loading .env file: {e}")

# Core data processing imports with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless backend; required for Streamlit Cloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    if os.environ.get("STREAMLIT_RUN") != "1":
        print("✅ Core data libraries loaded successfully")
except ImportError as e:
    print(f"❌ Error importing data libraries: {e}")
    print("\n🔧 To fix this issue, please run one of the following:")
    print("   pip uninstall numpy pandas -y")
    print("   pip install numpy==1.24.3")
    print("   pip install pandas==1.5.3")
    print("   pip install -r requirements.txt")
    if os.environ.get("STREAMLIT_RUN") != "1":
        input("Press Enter to exit...")
    sys.exit(1)

# File processing imports
try:
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract
    import requests
except ImportError as e:
    print(f"Error importing file processing libraries: {e}")
    print("Please run: pip install PyPDF2 python-docx Pillow pytesseract requests")
    sys.exit(1)

# Optional: streamlit is only needed for the UI module. We import it lazily
# inside _make_api_call_with_retry so Agent.py remains usable headless.
try:
    import streamlit as st  # noqa: F401
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# OpenCode Zen client
# ---------------------------------------------------------------------------

ZEN_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "minimax-m3-free"


def _get_api_key() -> Optional[str]:
    """
    Resolve the API key from, in order:
      1. st.secrets["OPENCODE_ZEN_API_KEY"]  (Streamlit Cloud)
      2. OPENCODE_ZEN_API_KEY env var
      3. TOGETHER_API_KEY env var (legacy)
    """
    # 1. Streamlit secrets (works only when running under Streamlit)
    if _ST_AVAILABLE:
        try:
            if "OPENCODE_ZEN_API_KEY" in st.secrets:
                return st.secrets["OPENCODE_ZEN_API_KEY"]
        except Exception:
            pass
    # 2. Environment
    return os.getenv("OPENCODE_API_KEY") or os.getenv("TOGETHER_API_KEY")


def _chat_complete(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.3,
    timeout: int = 120,
) -> str:
    """Single chat-completions call against OpenCode Zen. Returns the text."""
    url = f"{ZEN_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Raise on HTTP error so the retry helper can react
    if resp.status_code >= 400:
        # Try to extract a useful error message
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message") or err.get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
    data = resp.json()
    # OpenAI-style: choices[0].message.content
    try:
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, list):
            return " ".join(str(c) for c in content)
        return str(content) if content else ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {e} - {data}")


# ---------------------------------------------------------------------------
# DocumentAnalystAgent
# ---------------------------------------------------------------------------

class DocumentAnalystAgent:
    """
    An intelligent document analysis agent that can process multiple file formats,
    perform data analysis, generate visualizations, and answer questions.

    The agent is intentionally UI-agnostic: instantiate it with an API key and
    call its methods. The Streamlit UI lives in app.py.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):

        if not api_key:
            raise ValueError(
                "API key is required. Set OPENCODE_API_KEY in your environment, "
                "in .env, or in Streamlit Cloud secrets."
            )
        self.api_key = api_key
        self.model = model
        self.document_content: Dict[str, Dict[str, Any]] = {}
        self.data_frames: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    # ---- File extraction ----------------------------------------------------

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

    # ---- Document processing ------------------------------------------------

    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Process a document based on its type and extract relevant information.

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

            return result

        except Exception as e:
            result['content'] = f"Error processing file: {str(e)}"
            return result

    # ---- Analysis -----------------------------------------------------------

    def perform_data_analysis(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis on structured data.
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
        Create various visualizations for the data.
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
                plt.close(fig)
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

                fig, ax_array = plt.subplots(1, n_plots, figsize=(15, 5), squeeze=False)
                axes = ax_array.flatten()

                for i, col in enumerate(numeric_columns[:n_plots]):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')

                plt.tight_layout()
                box_path = os.path.join(output_dir, 'box_plots.png')
                plt.savefig(box_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
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
                plt.close(fig)
                visualization_paths.append(bar_path)

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

        return visualization_paths

    # ---- LLM interactions ---------------------------------------------------

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

    def answer_question(self, question: str, context_files: Optional[List[str]] = None) -> str:
        """
        Answer a question based on the processed documents.
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
        """Generate a comprehensive analysis report for a file"""
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

    # ---- API helper ---------------------------------------------------------

    def _make_api_call_with_retry(
        self,
        prompt: str,
        max_tokens: int = 500,
        max_retries: int = 3,
    ) -> str:
        """Make API call with retry logic and exponential backoff"""
        # Pull user-tweakable settings from Streamlit session state when available
        if _ST_AVAILABLE:
            try:
                if hasattr(st, "session_state"):
                    max_tokens = getattr(st.session_state, "max_tokens", max_tokens)
                    max_retries = getattr(st.session_state, "max_retries", max_retries)
                    temperature = getattr(st.session_state, "temperature", 0.3)
                else:
                    temperature = 0.3
            except Exception:
                temperature = 0.3
        else:
            temperature = 0.3

        messages = [{"role": "user", "content": prompt}]
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                return _chat_complete(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                # 429 / rate-limit → back off
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    if os.environ.get("STREAMLIT_RUN") != "1":
                        print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        return f"Rate limit exceeded. Please try again in a few minutes. Error: {error_str}"
                else:
                    # Non-rate-limit error: don't retry, surface to user
                    return f"API Error: {error_str}"

        return f"Maximum retries exceeded. Last error: {last_error}"
