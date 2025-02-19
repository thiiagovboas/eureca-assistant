from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ContextManager:
    def __init__(self) -> None:
        """
        Inicializa o gerenciador de contexto com estruturas vazias.
        """
        self.context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.empresa_info: Optional[Dict[str, Any]] = None
        self.last_update: datetime = datetime.utcnow()
        self.messages: List[BaseMessage] = []  # Nova lista para mensagens do LangChain

    def update_empresa_context(self, empresa_data: Dict[str, Any]) -> None:
        """
        Atualiza o contexto com informações da empresa.

        Args:
            empresa_data (Dict[str, Any]): Dicionário contendo dados da empresa
                Deve conter:
                - nome_empresa: str
                - setor: str
                - num_funcionarios: int
                - possui_programa: bool
                - dados_adicionais: Dict[str, Any] (opcional)
        """
        try:
            # Validação dos campos obrigatórios
            required_fields = ['nome_empresa', 'setor', 'num_funcionarios', 'possui_programa']
            missing_fields = [field for field in required_fields if field not in empresa_data]
            
            if missing_fields:
                raise ValueError(f"Campos obrigatórios ausentes: {', '.join(missing_fields)}")

            # Atualiza empresa_info com cópia dos dados para evitar referências mutáveis
            self.empresa_info = empresa_data.copy()
            
            # Atualiza o contexto com informações processadas
            self.context.update({
                'nome_empresa': empresa_data['nome_empresa'],
                'setor': empresa_data['setor'],
                'num_funcionarios': empresa_data['num_funcionarios'],
                'possui_programa': empresa_data['possui_programa'],
                'porte': self._categorize_company_size(empresa_data['num_funcionarios']),
                'stage': 'experiente' if empresa_data['possui_programa'] else 'iniciante',
                'last_update': datetime.utcnow().isoformat()
            })

            # Adiciona dados adicionais se existirem
            if 'dados_adicionais' in empresa_data:
                self.context.update(empresa_data['dados_adicionais'])

        except Exception as e:
            raise ValueError(f"Erro ao atualizar contexto da empresa: {str(e)}")

    def _categorize_company_size(self, num_funcionarios: int) -> str:
        """
        Categoriza o porte da empresa com base no número de funcionários.

        Args:
            num_funcionarios (int): Número total de funcionários da empresa

        Returns:
            str: Categoria do porte da empresa ('micro', 'pequeno', 'médio' ou 'grande')
        """
        if not isinstance(num_funcionarios, (int, float)):
            raise ValueError("Número de funcionários deve ser um número")
            
        if num_funcionarios < 20:
            return 'micro'
        elif num_funcionarios < 100:
            return 'pequeno'
        elif num_funcionarios < 500:
            return 'médio'
        return 'grande'

    def add_conversation_entry(self, question: str, answer: str) -> None:
        """
        Adiciona uma nova entrada no histórico da conversação.

        Args:
            question (str): Pergunta do usuário
            answer (str): Resposta do assistente
        """
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("Pergunta e resposta devem ser strings")

        # Adiciona à lista de mensagens do LangChain
        self.messages.append(HumanMessage(content=question.strip()))
        self.messages.append(AIMessage(content=answer.strip()))

        # Mantém o formato antigo para compatibilidade
        entry = {
            'question': f"Human: {question.strip()}",
            'answer': f"Assistant: {answer.strip()}",
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.conversation_history.append(entry)
        self.last_update = datetime.utcnow()

    def get_messages(self) -> List[BaseMessage]:
        """
        Retorna as mensagens no formato do LangChain.

        Returns:
            List[BaseMessage]: Lista de mensagens do LangChain
        """
        return self.messages

    def get_recent_history(self, limit: int = 5) -> Tuple[List[Dict[str, str]], List[BaseMessage]]:
        """
        Retorna as entradas mais recentes do histórico da conversação.

        Args:
            limit (int, optional): Número máximo de entradas a retornar. Defaults to 5.

        Returns:
            Tuple[List[Dict[str, str]], List[BaseMessage]]: 
                - Lista das últimas conversas no formato antigo
                - Lista das últimas mensagens no formato LangChain
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Limite deve ser um número inteiro positivo")
            
        return (
            self.conversation_history[-limit:],
            self.messages[-(limit*2):]  # *2 porque cada interação tem 2 mensagens
        )

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do contexto atual.

        Returns:
            Dict[str, Any]: Resumo do contexto incluindo informações da empresa
                e estatísticas da conversação
        """
        if not self.empresa_info:
            return {
                'error': 'Contexto da empresa não inicializado',
                'num_interactions': 0,
                'last_interaction': None,
                'context_age': 0
            }

        return {
            'empresa_info': self.empresa_info,
            'num_interactions': len(self.conversation_history),
            'last_interaction': self.conversation_history[-1] if self.conversation_history else None,
            'context_age': int((datetime.utcnow() - self.last_update).total_seconds() // 60),  # em minutos
            'porte': self.context.get('porte'),
            'stage': self.context.get('stage')
        }

    def clear_history(self) -> None:
        """
        Limpa o histórico de conversação mantendo as informações da empresa.
        """
        self.conversation_history = []
        self.messages = []  # Limpa também as mensagens do LangChain
        self.last_update = datetime.utcnow()

    def export_context(self) -> Dict[str, Any]:
        """
        Exporta todo o contexto e histórico para formato serializável.

        Returns:
            Dict[str, Any]: Dados completos do contexto e histórico
        """
        return {
            'context': self.context,
            'empresa_info': self.empresa_info,
            'conversation_history': self.conversation_history,
            'last_update': self.last_update.isoformat(),
            'metadata': {
                'export_time': datetime.utcnow().isoformat(),
                'num_interactions': len(self.conversation_history),
                'context_version': '1.1'
            }
        }