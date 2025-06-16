'''
It takes a while to answer complex files,
If my API key is not working, you can use your own API key.
Change the API key:
Use Ctrl+f to find(DEFAULT_API_KEY = )
'''


import os
import io
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# File processing imports
import PyPDF2
import docx
from PIL import Image
import pytesseract
import requests
from together import Together

# Optional UI imports (uncomment if you want to use Streamlit)
import streamlit as st
import subprocess
import sys
import webbrowser
from threading import Timer

class DocumentAnalystAgent:
    """
    An intelligent document analysis agent that can process multiple file formats,
    perform data analysis, generate visualizations, and answer questions.
    """
    
    def __init__(self, api_key: str):
        
        self.client = Together(api_key=api_key)
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
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
                fig, axes = plt.subplots(1, min(3, len(numeric_columns)), figsize=(15, 5))
                if len(numeric_columns) == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_columns[:3]):
                    if i < len(axes):
                        df.boxplot(column=col, ax=axes[i])
                        axes[i].set_title(f'Box Plot of {col}')
                
                plt.tight_layout()
                box_path = os.path.join(output_dir, 'box_plots.png')
                plt.savefig(box_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(box_path)
            
            # 4. Bar charts for categorical columns
            if len(categorical_columns) > 0:
                fig, axes = plt.subplots(1, min(2, len(categorical_columns)), figsize=(15, 6))
                if len(categorical_columns) == 1:
                    axes = [axes]
                
                for i, col in enumerate(categorical_columns[:2]):
                    if i < len(axes) and df[col].nunique() <= 20:
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": report_prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
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

# Streamlit UI - Simplified and User-Friendly
def create_streamlit_ui():
    # Page configuration
    st.set_page_config(
        page_title="Document Analyst Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📊 Document Analyst Agent")
    st.markdown("**Upload documents and ask intelligent questions about them**")
    st.markdown("---")
    
    # Initialize agent with your API key
    DEFAULT_API_KEY = "91cd400c4b4f60727630ca1f6b2affcf4c584fc4362a6711bd0e7b92a3e02cfd"
    
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = DocumentAnalystAgent(DEFAULT_API_KEY)
            st.success("✅ Agent initialized and ready!")
        except Exception as e:
            st.error(f"❌ Error initializing agent: {str(e)}")
            return
    
    agent = st.session_state.agent
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 File Manager")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'jpg', 'jpeg', 'png'],
            help="Supported formats: PDF, DOCX, TXT, CSV, XLSX, Images"
        )
        
        # Show processed files
        if agent.document_content:
            st.subheader("📄 Processed Files")
            file_info = agent.get_file_info()
            for file_name, info in file_info.items():
                with st.expander(f"📄 {file_name}"):
                    st.write(f"**Type:** {info['type'].upper()}")
                    st.write(f"**Has Data:** {'Yes' if info['has_data'] else 'No'}")
                    st.write(f"**Summary:** {info['summary']}")
        
        # Clear all files button
        if agent.document_content and st.button("🗑️ Clear All Files"):
            agent.document_content.clear()
            agent.data_frames.clear()
            agent.analysis_results.clear()
            agent.conversation_history.clear()
            st.success("All files cleared!")
            st.experimental_rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload & Process")
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Process file
                    result = agent.process_document(temp_path, uploaded_file.name)
                    
                    # Show processing result
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    
                    with st.expander(f"📋 Summary - {uploaded_file.name}"):
                        st.write(result['summary'])
                    
                    # If it's structured data, show analysis
                    if uploaded_file.name in agent.data_frames:
                        df = agent.data_frames[uploaded_file.name]
                        
                        with st.expander(f"📊 Data Preview - {uploaded_file.name}"):
                            st.dataframe(df.head(10))
                            
                            # Basic info
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Rows", df.shape[0])
                                st.metric("Columns", df.shape[1])
                            with col_b:
                                st.write("**Column Types:**")
                                st.write(df.dtypes.to_dict())
                        
                        # Perform analysis
                        analysis = agent.perform_data_analysis(df, uploaded_file.name)
                        
                        # Create and show visualizations
                        viz_paths = agent.create_visualizations(df, uploaded_file.name)
                        if viz_paths:
                            with st.expander(f"📈 Visualizations - {uploaded_file.name}"):
                                for viz_path in viz_paths:
                                    if os.path.exists(viz_path):
                                        st.image(viz_path, use_column_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All files processed!")
            progress_bar.empty()
            status_text.empty()
    
    with col2:
        st.header("💬 Ask Questions")
        
        if not agent.document_content:
            st.info("👆 Upload documents first to start asking questions!")
        else:
            # Question input
            question = st.text_area(
                "What would you like to know about your documents?",
                placeholder="e.g., What are the key insights from this data? What patterns do you see? Summarize the main points.",
                height=100
            )
            
            # Ask button
            if st.button("🔍 Get Answer", type="primary"):
                if question.strip():
                    with st.spinner("🤔 Thinking..."):
                        try:
                            answer = agent.answer_question(question)
                            st.success("💡 **Answer:**")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please enter a question!")
            
            # Conversation history
            if agent.conversation_history:
                st.subheader("💭 Conversation History")
                for i, item in enumerate(reversed(agent.conversation_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(agent.conversation_history)-i}: {item['question'][:50]}..."):
                        st.write(f"**Q:** {item['question']}")
                        st.write(f"**A:** {item['answer']}")
            
            # Quick questions for data files
            if agent.data_frames:
                st.subheader("🚀 Quick Questions")
                quick_questions = [
                    "What are the key statistics of this dataset?",
                    "What patterns or trends do you see in the data?",
                    "Are there any missing values or data quality issues?",
                    "What insights can you derive from this data?",
                    "What are the correlations between different variables?"
                ]
                
                for q in quick_questions:
                    if st.button(f"❓ {q}", key=f"quick_{q}"):
                        with st.spinner("🤔 Analyzing..."):
                            try:
                                answer = agent.answer_question(q)
                                st.success("💡 **Answer:**")
                                st.write(answer)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

def check_streamlit_server(port=8501):
    """Check if Streamlit server is running"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=2)
        return response.status_code == 200
    except:
        return False

def launch_with_browser_delay(url, delay=3):
    """Open browser after delay"""
    def open_browser():
        webbrowser.open(url)
        print(f"🌐 Opened {url} in browser")
    
    Timer(delay, open_browser).start()

def smart_streamlit_launch():
    """Smart Streamlit launcher with server checking"""
    port = 8501
    url = f"http://localhost:{port}"
    
    # Check if server is already running
    if check_streamlit_server(port):
        print(f"✅ Streamlit server already running at {url}")
        webbrowser.open(url)
        return
    
    try:
        print("🚀 Starting Document Analyst Agent...")
        
        script_path = os.path.abspath(__file__)
        
        # Schedule browser opening
        launch_with_browser_delay(url, 4)
        
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"📊 Launching Streamlit at {url}")
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
        import streamlit as st
        if hasattr(st, 'runtime') and st.runtime.exists():
            create_streamlit_ui()
        else:
            smart_streamlit_launch()
    except:
        smart_streamlit_launch()