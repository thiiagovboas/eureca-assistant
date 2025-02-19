from typing import List, Dict, Any, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import re

class ConversationProcessor:
    def __init__(self):
        self.question_patterns = {
            'multi_part': r'\?.*\?',  # Detecta múltiplas perguntas
            'numerical': r'\d+',       # Detecta números
            'comparison': r'diferença|versus|comparação|entre',  # Detecta comparações
            'requirement': r'preciso|necessário|obrigatório',    # Detecta requisitos
            'legal': r'lei|artigo|legislação|clt',  # Detecta referências legais
            'doubt': r'como|qual|quando|onde|por que|porque|quem|quanto',  # Perguntas comuns
        }

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Processa a pergunta para identificar componentes importantes e contexto.

        Args:
            question (str): A pergunta do usuário

        Returns:
            Dict[str, Any]: Análise detalhada da pergunta
        """
        parts = {
            'has_multiple_questions': bool(re.search(self.question_patterns['multi_part'], question)),
            'contains_numbers': bool(re.search(self.question_patterns['numerical'], question)),
            'is_comparison': bool(re.search(self.question_patterns['comparison'], question)),
            'is_requirement': bool(re.search(self.question_patterns['requirement'], question)),
            'has_legal_reference': bool(re.search(self.question_patterns['legal'], question, re.IGNORECASE)),
            'is_question': bool(re.search(self.question_patterns['doubt'], question, re.IGNORECASE)),
            'sub_questions': self._split_questions(question),
            'complexity': self._evaluate_complexity(question),
            'keywords': self._extract_keywords(question)
        }
        return parts

    def _split_questions(self, text: str) -> List[str]:
        """
        Divide texto em múltiplas perguntas se houver.

        Args:
            text (str): Texto a ser dividido

        Returns:
            List[str]: Lista de perguntas individuais
        """
        # Primeiro, trata casos especiais com "e" ou "ou"
        text = re.sub(r'\s+e\s+(?=como|qual|quando|onde|por que|porque|quem|quanto)', '? ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+ou\s+(?=como|qual|quando|onde|por que|porque|quem|quanto)', '? ', text, flags=re.IGNORECASE)
        
        # Depois divide por "?"
        return [q.strip() + '?' for q in text.split('?') if q.strip()]

    def _evaluate_complexity(self, question: str) -> str:
        """
        Avalia a complexidade da pergunta.

        Args:
            question (str): A pergunta a ser avaliada

        Returns:
            str: Nível de complexidade ('simples', 'média' ou 'complexa')
        """
        score = 0
        
        # Critérios de complexidade
        if self.question_patterns['has_multiple_questions']:
            score += 2
        if self.question_patterns['is_comparison']:
            score += 2
        if len(question.split()) > 20:
            score += 1
        if self.question_patterns['legal']:
            score += 1
        if len(self._split_questions(question)) > 1:
            score += 2

        if score <= 2:
            return 'simples'
        elif score <= 4:
            return 'média'
        return 'complexa'

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extrai palavras-chave relevantes da pergunta.

        Args:
            text (str): Texto para extrair palavras-chave

        Returns:
            List[str]: Lista de palavras-chave encontradas
        """
        # Palavras-chave comuns em perguntas sobre aprendizagem
        keywords = [
            'aprendiz', 'contrato', 'idade', 'salário', 'curso',
            'escola', 'horário', 'férias', 'direitos', 'deveres',
            'cota', 'contratação', 'rescisão', 'benefícios'
        ]
        
        found_keywords = []
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
                found_keywords.append(keyword)
                
        return found_keywords

    def process_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Processa uma lista de mensagens do LangChain para análise.

        Args:
            messages (List[BaseMessage]): Lista de mensagens do LangChain

        Returns:
            List[Dict[str, Any]]: Lista de análises das mensagens
        """
        analyses = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                analysis = self.process_question(msg.content)
                analyses.append({
                    'content': msg.content,
                    'type': 'human',
                    'analysis': analysis
                })
            elif isinstance(msg, AIMessage):
                analyses.append({
                    'content': msg.content,
                    'type': 'ai',
                    'length': len(msg.content.split())
                })
                
        return analyses