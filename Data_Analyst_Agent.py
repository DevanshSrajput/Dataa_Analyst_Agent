'''
If for privacy reason you don't trust the app use your own API key.
Change the API key at ".env"
If you can`t mail me at dksdevansh@gmail.com (I'll send you the key)
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

from typing import Dict, List, Any, Optional, Union

# File processing imports
try:
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract
    import requests
    from together import Together
except ImportError as e:
    print(f"Error importing file processing libraries: {e}")
    print("Please run: pip install PyPDF2 python-docx Pillow pytesseract requests together")
    sys.exit(1)

# UI imports
try:
    import streamlit as st
    import subprocess
except ImportError as e:
    print(f"Error importing UI libraries: {e}")
    print("Please run: pip install streamlit")
    sys.exit(1)

class DocumentAnalystAgent:
    """
    An intelligent document analysis agent that can process multiple file formats,
    perform data analysis, generate visualizations, and answer questions.
    """
    
    def __init__(self, api_key: str):
        
        self.client = Together(api_key=api_key)
        # Using a more accessible model with higher rate limits
        self.model = "meta-llama/Llama-3.1-8B-Instruct-Turbo"
        self.document_content = {}
        self.data_frames = {}
        self.analysis_results = {}
        self.conversation_history = []
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
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
                if "rate limit" in error_str.lower() or "429" in error_str:
                    wait_time = (2 ** attempt) * 30  # Exponential backoff: 30s, 60s, 120s
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        return f"Rate limit exceeded. Please try again in a few minutes. Error: {error_str}"
                else:
                    # For non-rate-limit errors, don't retry
                    return f"API Error: {error_str}"
        
        return "Maximum retries exceeded. Please try again later."

def create_streamlit_ui():
    """Create the Streamlit user interface"""
    # Page configuration with custom styling
    st.set_page_config(
        page_title="📊 AI Document Analyst | Smart Document Processing",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/DevanshSrajput/Dataa_Analyst_Agent',
            'Report a bug': "mailto:dksdevansh@gmail.com",
            'About': "AI-powered document analysis tool by Devansh Singh"
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
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Built by Devansh Singh | Powered by Meta Llama & Together AI | {st.session_state.theme_mode.title()} Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent with enhanced error handling
    api_key = os.getenv('TOGETHER_API_KEY')
    
    if not api_key:
        st.error("🔐 **API Key Required!** Please set your TOGETHER_API_KEY in the .env file")
        with st.expander("🔧 How to set up API Key"):
            st.markdown("""
            1. Get your API key from [Together AI](https://api.together.xyz/)
            2. Create a `.env` file in your project directory
            3. Add: `TOGETHER_API_KEY=your_api_key_here`
            4. Restart the application
            """)
        return
    
    if 'agent' not in st.session_state:
        with st.spinner("🚀 Initializing AI Agent..."):
            try:
                st.session_state.agent = DocumentAnalystAgent(api_key)
                st.success("✅ AI Agent ready for action!")
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
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
            current_key = os.getenv('TOGETHER_API_KEY')
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
            
            **📧 Support:** dksdevansh@gmail.com
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
        st.markdown("Built by Devansh Singh")
        st.markdown("Powered by Together AI")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📤 Upload & Process", "💬 AI Chat", "📊 Analytics", "⚙️ Settings"])
    
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
            current_key = os.getenv('TOGETHER_API_KEY')
            if current_key:
                st.success(f"✅ **Current API Key Status:** Active")
                st.text(f"🔐 Key Preview: {current_key[:12]}...{current_key[-8:]}")
                st.text(f"📅 Loaded from: Environment (.env file)")
            else:
                st.error("❌ **No API Key Found**")
                st.warning("Please set your TOGETHER_API_KEY in the .env file or use temporary override below.")
            
            # Temporary API Key Override
            st.markdown("#### 🔄 Temporary API Key Override")
            st.info("💡 This will override your .env API key for this session only")
            
            temp_api_key = st.text_input(
                "Enter Temporary API Key:",
                type="password",
                placeholder="Enter your Together AI API key here...",
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
            1. Visit [Together AI](https://api.together.xyz/)
            2. Sign up or log in to your account
            3. Navigate to API Keys section
            4. Create a new API key
            5. Copy and paste it here or in your .env file
            """)
            
            if st.button("🌐 Open Together AI", use_container_width=True):
                st.markdown("[🔗 Click here to visit Together AI](https://api.together.xyz/)")
        
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
        - Built by Devansh Singh
        - Powered by Together AI & Meta Llama
        - GitHub: [Dataa_Analyst_Agent](https://github.com/DevanshSrajput/Dataa_Analyst_Agent)
        - Email: dksdevansh@gmail.com
        
        **🚀 Features:**
        - Multi-format document processing
        - AI-powered analysis and insights
        - Interactive chat interface
        - Automatic data visualization
        - Comprehensive analytics dashboard
        """)

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