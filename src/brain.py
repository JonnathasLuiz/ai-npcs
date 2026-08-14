"""
Arquitetura de Rede Neural do Cérebro do NPC baseada em PPO (Actor-Critic).
Implementa o mecanismo de Multi-Head Attention para correlacionar o estado do NPC com o mapa.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    TOKEN_VOCAB_SIZE,
    NPC_STATE_SIZE,
    ACTION_SPACE_SIZE,
)


class NPCBrain(nn.Module):
    """
    Arquitetura de rede neural customizada para tomada de decisão de NPCs.

    Usa um cérebro do tipo Actor-Critic (PPO) integrado a uma camada de
    Multi-Head Attention. O NPC cruza o seu estado atual (Fome, Energia, etc.)
    com as assinaturas semânticas dos edifícios do mapa para decidir a melhor ação.
    """

    # Definição explícita das camadas e tipos
    state_mlp: nn.Sequential
    node_embedding: nn.Linear
    attention: nn.MultiheadAttention
    fusion_layer: nn.Sequential
    actor_head: nn.Linear
    critic_head: nn.Linear

    def __init__(self) -> None:
        """
        Inicializa e constrói as camadas da rede neural do cérebro do NPC.
        """
        super(NPCBrain, self).__init__()

        # --- VIA 1: Processamento de Estado Interno do NPC ---
        # Recebe os atributos de estado e projeta em um vetor de características latentes (dimensão 32)
        self.state_mlp = nn.Sequential(
            nn.Linear(NPC_STATE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # --- VIA 2: Processamento Espacial (Edifícios / Nós Semânticos) ---
        # Mapeia as assinaturas semânticas de entrada dos edifícios para uma representação latente (dimensão 32)
        self.node_embedding = nn.Linear(TOKEN_VOCAB_SIZE, 32)

        # --- CAMADA DE ATENÇÃO MULTI-HEAD ---
        # Correlaciona as necessidades atuais do NPC (Query) com o que cada edifício ao redor oferece (Key/Value)
        # Dimensão latente global de 32 com 2 cabeças de atenção independentes
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)

        # --- FUSÃO E TOMADA DE DECISÃO ---
        # Combina a informação do estado do NPC (32) com o vetor contextual obtido via atenção (32)
        # 32 + 32 = 64 dimensões de entrada
        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Cabeça do Ator (Actor): Define a distribuição de probabilidades das ações possíveis
        self.actor_head = nn.Linear(64, ACTION_SPACE_SIZE)

        # Cabeça do Crítico (Critic): Estima o valor (recompensa esperada) do estado atual
        self.critic_head = nn.Linear(64, 1)

    def forward(self, npc_state: torch.Tensor, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executa a passagem para frente (forward pass) / inferência do cérebro do NPC.

        Args:
            npc_state (torch.Tensor): Tensor de tamanho (Batch, NPC_STATE_SIZE) com as necessidades internas.
            nodes (torch.Tensor): Tensor de tamanho (Batch, MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE)
                contendo as assinaturas dos edifícios próximos.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - action_logits: Probabilidades não-normalizadas (logits) das ações (Batch, ACTION_SPACE_SIZE).
                - state_value: Estimativa do valor de estado pelo crítico (Batch, 1).
        """

        # 1. Processamento das necessidades internas (Query) -> tamanho (Batch, 32)
        q_state: torch.Tensor = self.state_mlp(npc_state)

        # 2. Processamento semântico dos edifícios (Keys/Values) -> tamanho (Batch, MAX_NODES_PER_QUERY, 32)
        k_nodes: torch.Tensor = self.node_embedding(nodes)

        # 3. Aplicação do mecanismo de atenção
        # Adiciona dimensão de sequência para a Query para atender ao formato de entrada da atenção
        # (Batch, 1, 32)
        q_state_unsqueeze: torch.Tensor = q_state.unsqueeze(1)

        # Aplicação da camada de atenção. attn_output possui tamanho (Batch, 1, 32)
        attn_output, _ = self.attention(
            query=q_state_unsqueeze,
            key=k_nodes,
            value=k_nodes
        )

        # Remove a dimensão extra da sequência temporal, voltando para (Batch, 32)
        context_vector: torch.Tensor = attn_output.squeeze(1)

        # 4. Fusão das características processadas
        # Concatena o estado de necessidade original e o contexto semântico do mapa -> (Batch, 64)
        fused: torch.Tensor = torch.cat((q_state, context_vector), dim=1)

        # Passa pela rede profunda de fusão para consolidar a representação do cenário
        features: torch.Tensor = self.fusion_layer(fused)

        # 5. Geração das saídas para política de decisão e julgamento de valor
        action_logits: torch.Tensor = self.actor_head(features)
        state_value: torch.Tensor = self.critic_head(features)

        return action_logits, state_value
