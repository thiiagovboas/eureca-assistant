from string import Template
from typing import Dict, List, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

class PromptTemplate:
    # Template principal com regras mais específicas
    MAIN_SYSTEM_PROMPT = Template("""Você é o assistente virtual da Eureca, especialista em Jovem Aprendiz.

REGRAS OBRIGATÓRIAS (SEMPRE SIGA ESTAS REGRAS):
1. NUNCA mencione leis, artigos ou base legal, a menos que EXPLICITAMENTE solicitado
2. NUNCA liste "próximos passos" ou qualquer tipo de lista numerada
3. NUNCA use formatos automáticos como "1.", "2.", etc.
4. NUNCA adicione informações não solicitadas
5. SEMPRE use o nome real da empresa: ${nome_empresa}
6. SEMPRE use o setor real da empresa: ${setor}
7. SEMPRE mantenha um tom amigável e consultivo
8. SEMPRE personalize as respostas para o contexto da empresa

SEU COMPORTAMENTO:
- Você é prestativo e focado em soluções
- Você entende profundamente sobre aprendizagem
- Você conhece as necessidades específicas de cada setor
- Você mantém o foco na pergunta atual
- Você evita termos técnicos desnecessários

ESTRUTURA DE RESPOSTA:
1. Comece reconhecendo o contexto da empresa
2. Responda à pergunta de forma direta
3. Personalize a informação para o setor
4. Termine com uma abertura para mais perguntas

O QUE EVITAR:
- NÃO use "Base legal:" ou similar
- NÃO use "Próximos passos:" ou similar
- NÃO cite artigos da CLT sem solicitação
- NÃO faça listas numeradas
- NÃO use linguagem muito formal""")

    # Template específico para saudações melhorado
    GREETING_TEMPLATE = Template("""Você é o assistente virtual da Eureca. 
    
CONTEXTO ESPECÍFICO:
Empresa: ${nome_empresa}
Setor: ${setor}
Status: ${possui_programa}

INSTRUÇÕES EXATAS:
Responda EXATAMENTE neste formato:
"Olá! Que bom ter você aqui! Sou o assistente da Eureca e estou aqui para ajudar a ${nome_empresa} com tudo relacionado à Lei de Aprendizagem. Vi que vocês são do setor de ${setor} e ${status_programa}. Como posso ajudar hoje?"

REGRAS CRÍTICAS:
- Use EXATAMENTE o formato acima
- NÃO adicione NADA além do texto especificado
- NÃO mencione leis ou artigos
- NÃO sugira próximos passos
- NÃO inclua informações adicionais""")

    @staticmethod
    def is_greeting(text: str) -> bool:
        """Verifica se o texto é uma saudação."""
        saudacoes = {
            'oi', 'olá', 'ola', 'hi', 'hello', 'ei', 
            'bom dia', 'boa tarde', 'boa noite', 'hey'
        }
        return text.lower().strip('!., ') in saudacoes

    @staticmethod
    def format_chat_history(messages: List[BaseMessage]) -> str:
        """
        Formata o histórico de mensagens do LangChain.

        Args:
            messages (List[BaseMessage]): Lista de mensagens do LangChain

        Returns:
            str: Histórico formatado
        """
        formatted_history = []
        
        for msg in messages:
            prefix = "Usuário: " if isinstance(msg, HumanMessage) else "Assistente: "
            formatted_history.append(f"{prefix}{msg.content}")
            
        return "\n".join(formatted_history)

    @staticmethod
    def generate_greeting(context: Dict[str, Any]) -> Union[str, List[BaseMessage]]:
        """
        Gera uma saudação inicial personalizada.

        Args:
            context (Dict[str, Any]): Contexto atual

        Returns:
            Union[str, List[BaseMessage]]: Mensagens para o LangChain ou string do template
        """
        possui_programa = context.get('possui_programa') == "Sim"
        status_programa = (
            "já possuem um programa de aprendizagem" 
            if possui_programa 
            else "ainda não possuem um programa de aprendizagem"
        )

        # Garante que temos valores válidos
        nome_empresa = context.get('nome_empresa')
        setor = context.get('setor')
        
        if not nome_empresa or not setor:
            # Fallback seguro se faltar informação
            greeting = """Olá! Que bom ter você aqui! Sou o assistente da Eureca e estou aqui para ajudar com tudo relacionado à Lei de Aprendizagem. Como posso ajudar hoje?"""
            return [SystemMessage(content=greeting)]

        # Cria o dicionário com todas as variáveis necessárias
        template_vars = {
            'nome_empresa': nome_empresa,
            'setor': setor,
            'possui_programa': context.get('possui_programa', 'Não informado'),
            'status_programa': status_programa
        }

        try:
            greeting = PromptTemplate.GREETING_TEMPLATE.substitute(template_vars)
            return [SystemMessage(content=greeting)]
        except KeyError as e:
            # Fallback seguro em caso de erro
            greeting = f"""Olá! Que bom ter você aqui! Sou o assistente da Eureca e estou aqui para ajudar a {nome_empresa} com tudo relacionado à Lei de Aprendizagem. Como posso ajudar hoje?"""
            return [SystemMessage(content=greeting)]

    @staticmethod
    def generate_prompt(
        context: Dict[str, Any], 
        question: str, 
        chat_history: Union[List[Dict[str, str]], List[BaseMessage]]
    ) -> Union[str, List[BaseMessage]]:
        """
        Gera prompt personalizado para a interação.

        Args:
            context (Dict[str, Any]): Contexto atual
            question (str): Pergunta do usuário
            chat_history: Histórico de chat (formato antigo ou LangChain)

        Returns:
            Union[str, List[BaseMessage]]: Prompt personalizado
        """
        # Verifica se é uma saudação
        if PromptTemplate.is_greeting(question):
            return PromptTemplate.generate_greeting(context)

        # Formata o histórico baseado no tipo
        if isinstance(chat_history[0], BaseMessage) if chat_history else False:
            chat_history_text = PromptTemplate.format_chat_history(chat_history[-6:])  # últimas 3 interações
        else:
            chat_history_text = "\n".join([
                f"Usuário: {h['question']}\nAssistente: {h['answer']}" 
                for h in chat_history[-3:]
            ]) if chat_history else ""

        # Garante que temos valores válidos
        nome_empresa = context.get('nome_empresa', 'Empresa')
        setor = context.get('setor', 'não especificado')

        try:
            # Cria o prompt do sistema usando o Template
            system_prompt = PromptTemplate.MAIN_SYSTEM_PROMPT.substitute({
                'nome_empresa': nome_empresa,
                'setor': setor
            })

            # Retorna lista de mensagens para o LangChain
            return [
                SystemMessage(content=system_prompt),
                *([SystemMessage(content=f"\nCONTEXTO ATUAL:\n{chat_history_text}")] if chat_history_text else []),
                HumanMessage(content=question)
            ]
        except KeyError as e:
            # Fallback seguro em caso de erro
            system_prompt = """Você é o assistente virtual da Eureca, especialista em Jovem Aprendiz."""
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]