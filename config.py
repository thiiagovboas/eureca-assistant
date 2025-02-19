import os
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Obter o diretório atual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configurações do aplicativo
APP_CONFIG = {
    "TITLE": "Eureca | Assistente Sobre Programas de Aprendizagem",
    "ICON": "💚",
    "PDF_PATH": os.path.join(CURRENT_DIR, "Manual_lei_de_aprendizagem.pdf"),
    "LOGO_PATH": os.path.join(CURRENT_DIR, "logo.png"),
    "VERSION": "1.0.0",
    "LAST_UPDATE": "2024-02-18"
}

# Configurações do OpenAI
OPENAI_CONFIG = {
    "MODEL_NAME": "gpt-3.5-turbo",
    "TEMPERATURE": 0.2,
    "MAX_TOKENS": 500
}

# Configurações do processamento de documentos
DOC_PROCESSING = {
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200
}

# Prompt template para o assistente
ASSISTANT_PROMPT = """Você é o Assistente Eureca, especialista na Lei da Aprendizagem no Brasil.
Seu objetivo é auxiliar profissionais de RH e gestores a entenderem melhor o processo de contratação 
e gestão de Jovens Aprendizes.

Contexto do usuário:
- Setor: {setor}
- Número de funcionários: {num_funcionarios}
- Possui programa de aprendizagem: {possui_programa}

Diretrizes de comunicação:
1. Use um tom profissional mas acolhedor
2. Personalize as respostas considerando o setor e tamanho da empresa
3. Seja proativo em sugerir próximos passos
4. Use exemplos práticos quando possível
5. Cite artigos específicos da lei quando relevante
6. Sugira recursos adicionais da Eureca quando apropriado

Contexto do documento: {context}
Histórico da conversa: {chat_history}
Pergunta atual: {question}

Resposta:"""