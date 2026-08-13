"""
Configurações e constantes globais para a simulação de economia com IA.
"""

# Máximo de edifícios/nós que o NPC consegue enxergar simultaneamente
MAX_NODES_PER_QUERY: int = 5

# Quantidade máxima de tokens de recursos na assinatura semântica (representação do recurso)
TOKEN_VOCAB_SIZE: int = 10

# Tamanho do estado interno do NPC (Fome, Energia, Dinheiro, Profissão)
NPC_STATE_SIZE: int = 4

# Quantidade total de ações discretas disponíveis no espaço de ação do NPC
# (Exemplo: 0 a 4 = interagir com nós específicos, 5 = permanecer parado)
ACTION_SPACE_SIZE: int = 6
