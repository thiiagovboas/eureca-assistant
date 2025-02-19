import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
import pickle
import hashlib
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página
try:
    st.set_page_config(
        page_title="Assistente Eureca",
        page_icon="💡",
        layout="centered"
    )
except Exception as e:
    logger.error(f"Erro na configuração da página: {e}")
    st.error("Erro na inicialização da página")

def get_file_hash(file_path: str) -> str:
    """Calcula o hash do arquivo para verificar mudanças"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Erro ao calcular hash do arquivo {file_path}: {e}")
        return None

@st.cache_resource(show_spinner=True)
def load_pdf_contents():
    logger.info("Iniciando carregamento dos PDFs")
    
    # Obtem o diretório atual do script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    pdf_paths = {
        "manual": os.path.join(current_dir, "Manual_lei_de_aprendizagem.pdf"),
        "boas_praticas": os.path.join(current_dir, "Boas_Práticas_na_Seleção_de_Jovens_Aprendizes.pdf"),
        "sobre": os.path.join(current_dir, "Sobre_Eureca.pdf")
    }
    
    # Verifica se os arquivos existem
    for name, path in pdf_paths.items():
        if not os.path.exists(path):
            logger.error(f"Arquivo não encontrado: {path}")
            st.error(f"Arquivo {name} não encontrado em: {path}")
            return None

    try:
        # Configuração do DocumentConverter
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
        )
        
        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # Inicializa barra de progresso
        progress_text = "Carregando documentos..."
        my_bar = st.progress(0, text=progress_text)
        
        documents = {}
        total_files = len(pdf_paths)
        
        for idx, (name, path) in enumerate(pdf_paths.items(), 1):
            try:
                logger.info(f"Processando documento: {name}")
                # Atualiza barra de progresso
                progress = idx/total_files
                my_bar.progress(progress, text=f"Processando {name}... {int(progress*100)}%")
                
                # Converte PDF
                conv_result = doc_converter.convert(path)
                markdown_content = conv_result.document.export_to_markdown()
                documents[name] = markdown_content
                
                logger.info(f"Documento {name} processado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao processar {name}: {e}")
                st.error(f"Erro ao processar {name}: {str(e)}")
                continue
        
        my_bar.empty()
        return documents if documents else None
        
    except Exception as e:
        logger.error(f"Erro no carregamento dos PDFs: {e}")
        st.error(f"Erro no carregamento dos documentos: {str(e)}")
        return None

def find_relevant_content(query: str, documents: dict) -> str:
    try:
        relevant_parts = []
        query_words = set(query.lower().split())
        
        for doc_name, content in documents.items():
            paragraphs = content.split('\n\n')
            scored_paragraphs = []
            
            for para in paragraphs:
                score = sum(para.lower().count(word) for word in query_words)
                if score > 0:
                    scored_paragraphs.append((score, para))
            
            scored_paragraphs.sort(reverse=True)
            relevant_parts.extend([p for _, p in scored_paragraphs[:2]])
        
        context = "\n\n".join(relevant_parts)
        return context[:4000]
    except Exception as e:
        logger.error(f"Erro ao encontrar conteúdo relevante: {e}")
        return ""

# Verifica chave da API
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("Chave da API OpenAI não encontrada")
    st.error("⚠️ Chave da API OpenAI não encontrada! Verifique seu arquivo .env")
    st.stop()

# Carrega os documentos com tratamento de erro
try:
    with st.spinner('Carregando documentos...'):
        documents = load_pdf_contents()
    
    if documents is None:
        logger.error("Falha ao carregar documentos")
        st.error("Não foi possível carregar os documentos. Verifique os arquivos e tente novamente.")
        st.stop()

except Exception as e:
    logger.error(f"Erro durante o carregamento: {e}")
    st.error("Erro durante o carregamento dos documentos")
    st.stop()

# Inicializa o cliente OpenAI
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    logger.error(f"Erro ao inicializar cliente OpenAI: {e}")
    st.error("Erro ao inicializar o cliente OpenAI")
    st.stop()

# Interface do usuário
st.title("💡 Assistente Eureca")
st.markdown("Especializado em Lei de Aprendizagem e processos de seleção")

# Inicialização do histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Olá! Sou o assistente especializado da Eureca. Posso ajudar você com informações sobre:\n\n"
                      "📚 Lei de Aprendizagem\n"
                      "🎯 Boas práticas na seleção de jovens aprendizes\n"
                      "ℹ️ Informações sobre a Eureca\n\n"
                      "Como posso ajudar você hoje?"
        }
    ]

# Exibe histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do usuário
if prompt := st.chat_input("Faça sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    relevant_context = find_relevant_content(prompt, documents)
    
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""Você é um assistente especializado da Eureca, focado em:
                        1. Lei de Aprendizagem
                        2. Boas práticas na seleção de jovens aprendizes
                        3. Informações sobre a Eureca
                        
                        Use APENAS as informações do contexto fornecido para responder às perguntas.
                        Se a informação não estiver no contexto, diga que não tem essa informação específica.
                        
                        Contexto relevante dos documentos:
                        {relevant_context}"""
                    },
                    *st.session_state.messages
                ],
                stream=True,
                temperature=0.7
            )
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            logger.error(f"Erro na resposta do OpenAI: {e}")
            st.error(f"Erro ao gerar resposta: {str(e)}")

# Rodapé
st.markdown("---")
st.markdown("💡 *Pergunte sobre a Lei de Aprendizagem, processos de seleção ou sobre a Eureca!*")
