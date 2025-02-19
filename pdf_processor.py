from typing import Optional, List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from datetime import datetime, timedelta
import hashlib

class PDFProcessor:
    def __init__(self):
        """
        Inicializa o processador de PDFs com configurações padrão.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_files = {
            'manual': os.path.join(current_dir, "Manual_lei_de_aprendizagem.pdf"),
            'boas_praticas': os.path.join(current_dir, "Boas_Práticas_na_Seleção_de_Jovens_Aprendizes.pdf"),
            'sobre_eureca': os.path.join(current_dir, "Sobre_Eureca.pdf")
        }
        self.vectorstore: Optional[FAISS] = None
        self.last_processed: Optional[datetime] = None
        self.file_hashes: Dict[str, str] = {}
        
        # Configurações do processamento
        self.chunk_settings = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'length_function': len
        }
        
        # Tempo máximo de cache (12 horas)
        self.cache_duration = timedelta(hours=12)

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calcula o hash MD5 de um arquivo.

        Args:
            file_path (str): Caminho do arquivo

        Returns:
            str: Hash MD5 do arquivo
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _should_reprocess(self) -> bool:
        """
        Verifica se os PDFs precisam ser reprocessados.

        Returns:
            bool: True se precisar reprocessar
        """
        if not self.vectorstore or not self.last_processed:
            return True

        # Verifica tempo de cache
        if datetime.utcnow() - self.last_processed > self.cache_duration:
            return True

        # Verifica mudanças nos arquivos
        for file_path in self.pdf_files.values():
            if not os.path.exists(file_path):
                return True
            
            current_hash = self._calculate_file_hash(file_path)
            if file_path not in self.file_hashes or self.file_hashes[file_path] != current_hash:
                return True

        return False

    def _process_single_pdf(self, file_path: str, file_type: str) -> List[Document]:
        """
        Processa um único arquivo PDF.

        Args:
            file_path (str): Caminho do arquivo
            file_type (str): Tipo do arquivo (manual, boas_praticas, etc.)

        Returns:
            List[Document]: Lista de documentos processados
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata.update({
                'source_type': file_type,
                'filename': file_path,
                'processed_at': datetime.utcnow().isoformat(),
                'file_hash': self._calculate_file_hash(file_path)
            })
            
        return documents

    def process_pdf(self) -> bool:
        """
        Processa todos os PDFs e cria/atualiza o vectorstore.

        Returns:
            bool: True se o processamento foi bem sucedido
        """
        try:
            # Verifica se pode usar cache
            if not self._should_reprocess() and 'vectorstore' in st.session_state:
                self.vectorstore = st.session_state.vectorstore
                return True

            # Verifica existência dos arquivos
            file_status = {
                name: os.path.exists(file_path)
                for name, file_path in self.pdf_files.items()
            }
            
            missing_files = [name for name, exists in file_status.items() if not exists]
            
            if missing_files:
                st.error(f"Arquivos não encontrados: {', '.join(missing_files)}")
                return False

            # Processa todos os documentos
            all_documents = []
            self.file_hashes = {}
            
            for file_type, file_path in self.pdf_files.items():
                try:
                    documents = self._process_single_pdf(file_path, file_type)
                    all_documents.extend(documents)
                    self.file_hashes[file_path] = self._calculate_file_hash(file_path)
                    
                except Exception as e:
                    st.error(f"Erro ao processar {file_path}: {str(e)}")
                    return False

            # Divide os documentos
            text_splitter = RecursiveCharacterTextSplitter(
                **self.chunk_settings
            )
            splits = text_splitter.split_documents(all_documents)

            # Cria embeddings e vectorstore
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Atualiza cache
            st.session_state.vectorstore = self.vectorstore
            self.last_processed = datetime.utcnow()
            st.session_state.last_pdf_update = self.last_processed
            
            return True

        except Exception as e:
            st.error(f"Erro ao processar PDFs: {str(e)}")
            return False

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[Document]:
        """
        Recupera os chunks mais relevantes para uma query.

        Args:
            query (str): Query de busca
            k (int): Número de chunks a retornar

        Returns:
            List[Document]: Lista de documentos relevantes
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore não inicializado")
            
        return self.vectorstore.similarity_search(query, k=k)

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Retorna o status atual do processamento.

        Returns:
            Dict[str, Any]: Status do processamento
        """
        return {
            'last_processed': self.last_processed,
            'files_processed': list(self.file_hashes.keys()) if self.file_hashes else [],
            'cache_valid': not self._should_reprocess() if self.vectorstore else False,
            'vectorstore_initialized': self.vectorstore is not None
        }
