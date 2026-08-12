import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Configurações da simulação para pré-alocação (Zero-Padding)
MAX_NODES_PER_QUERY = 5     # Máximo de edifícios que o NPC enxerga por vez
TOKEN_VOCAB_SIZE = 10       # Quantidade máxima de tokens de recursos (expandível)
NPC_STATE_SIZE = 4          # Fome, Energia, Dinheiro, Profissão
ACTION_SPACE_SIZE = 6       # Ações possíveis (Ex: 0 a 4 = Interagir com nós, 5 = Ficar parado)

class EconomicEnv(gym.Env):
    """
    Ambiente Headless para treinar os NPCs usando Aprendizado por Reforço.
    Não possui gráficos. Roda milhões de ticks por segundo focado na matemática.
    """
    def __init__(self):
        super(EconomicEnv, self).__init__()
        
        # Define o que a IA pode FAZER (Output)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        
        # Define o que a IA pode VER (Input)
        # Usamos um Dict space para separar o estado interno do NPC do mundo externo (Nós)
        self.observation_space = spaces.Dict({
            "npc_state": spaces.Box(low=0, high=1, shape=(NPC_STATE_SIZE,), dtype=np.float32),
            "nodes": spaces.Box(low=0, high=1, shape=(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE), dtype=np.float32)
        })
        
        # Variáveis internas para simulação
        self.current_step = 0
        self.max_steps = 1000

        # --- NOVO: Sistema de Recompensa Customizável ---
        # Lista de callbacks que definem o que gera recompensa ou punição.
        self.reward_callbacks = []
        
        # Guardamos o estado anterior para calcular deltas (ex: se o dinheiro aumentou)
        self._current_npc_state = None

    def add_reward_rule(self, condition_callback, reward_callback):
        """
        Funciona exatamente como a sua ideia de 'agent.assert(status, callback)'.
        - condition_callback: avalia o status. Recebe (old_state, new_state, action) -> retorna Boolean.
        - reward_callback: aplica a pontuação. Recebe (old_state, new_state, action) -> retorna Float.
        """
        self.reward_callbacks.append((condition_callback, reward_callback))

    def reset(self, seed=None):
        """Reinicia o episódio quando o NPC morre ou o tempo acaba."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Gera um estado inicial aleatório para o treino
        self._current_npc_state = np.random.rand(NPC_STATE_SIZE).astype(np.float32)
        nodes = np.random.rand(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE).astype(np.float32)
        
        return {"npc_state": self._current_npc_state, "nodes": nodes}, {}

    def step(self, action):
        """Executa a ação escolhida pela IA e calcula a recompensa (Salário da IA)."""
        self.current_step += 1
        
        # AQUI ENTRARIA A LÓGICA DO SEU JOGO:
        # Exemplo: Se action == 0 (Ir para Fazenda e trabalhar), aumentar dinheiro, reduzir energia.
        
        # Gera o próximo estado (mock para o exemplo)
        next_npc_state = np.random.rand(NPC_STATE_SIZE).astype(np.float32)
        next_nodes = np.random.rand(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE).astype(np.float32)
        
        # --- NOVO: Avaliação Dinâmica de Recompensa ---
        reward = 0.0
        # Itera sobre todas as regras que foram injetadas neste agente
        for condition_cb, reward_cb in self.reward_callbacks:
            # Se o status (condition_cb) for verdadeiro, aplica o callback de recompensa
            if condition_cb(self._current_npc_state, next_npc_state, action):
                reward += reward_cb(self._current_npc_state, next_npc_state, action)
        
        # Verifica se o episódio acabou (Ex: NPC morreu de fome ou atingiu max_steps)
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Atualiza a memória de estado do NPC para o próximo tick
        self._current_npc_state = next_npc_state
        
        obs = {"npc_state": next_npc_state, "nodes": next_nodes}
        
        return obs, reward, terminated, truncated, {}

class NPCBrain(nn.Module):
    """
    Arquitetura PPO (Actor-Critic) customizada com Multi-Head Attention.
    """
    def __init__(self):
        super(NPCBrain, self).__init__()
        
        # --- VIA 1: Processamento do Estado do NPC ---
        self.state_mlp = nn.Sequential(
            nn.Linear(NPC_STATE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 32) # Transforma os 4 atributos num vetor rico de tamanho 32
        )
        
        # --- VIA 2: Processamento Espacial (Nós Semânticos) ---
        self.node_embedding = nn.Linear(TOKEN_VOCAB_SIZE, 32)
        
        # Camada de Atenção (embed_dim=32, num_heads=2)
        # batch_first=True significa que os tensores entrarão no formato (Batch, Nodes, Embedding)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        
        # --- FUSÃO E TOMADA DE DECISÃO ---
        # 32 (Via 1) + 32 (Via 2) = 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Cabeça do Ator (Política - Qual ação tomar?)
        self.actor_head = nn.Linear(64, ACTION_SPACE_SIZE)
        
        # Cabeça do Crítico (Valor - Quão boa é essa situação?)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, npc_state, nodes):
        """Passagem dos dados pela rede (Inference)."""
        
        # 1. Processa o estado interno do NPC (Query)
        # npc_state tem formato (Batch, 4) -> vira (Batch, 32)
        q_state = self.state_mlp(npc_state) 
        
        # 2. Processa as assinaturas dos nós (Keys e Values)
        # nodes tem formato (Batch, 5, 10) -> vira (Batch, 5, 32)
        k_nodes = self.node_embedding(nodes) 
        
        # 3. Camada de Atenção (O NPC cruza o que ele quer com o que o mundo oferece)
        # A Query precisa ter a dimensão temporal/sequência, então adicionamos uma dimensão extra
        # q_state_unsqueeze vira (Batch, 1, 32)
        q_state_unsqueeze = q_state.unsqueeze(1)
        
        # Aplica a atenção. Resultado: attn_output tem formato (Batch, 1, 32)
        attn_output, _ = self.attention(query=q_state_unsqueeze, key=k_nodes, value=k_nodes)
        
        # Removemos a dimensão extra do resultado (Batch, 32)
        context_vector = attn_output.squeeze(1)
        
        # 4. Fusão das Vias
        # Junta o estado interno (32) com o contexto do mapa (32) -> (Batch, 64)
        fused = torch.cat((q_state, context_vector), dim=1)
        
        # Passa pela rede de decisão
        features = self.fusion_layer(fused)
        
        # 5. Saídas (Actor e Critic)
        action_logits = self.actor_head(features) # (Batch, 6)
        state_value = self.critic_head(features)  # (Batch, 1)
        
        return action_logits, state_value

if __name__ == "__main__":
    print("--- Inicializando Arquitetura de IA para Economia ---")
    
    # 1. Instancia o Ambiente e a Rede
    env = EconomicEnv()
    brain = NPCBrain()
    
    # --- NOVO: Configurando as Regras de Recompensa Customizáveis ---
    # Lembrete dos índices em nosso array: 0=Fome, 1=Energia, 2=Dinheiro, 3=Profissão

    # Regra 1: Punição por ficar parado (Ação 5)
    env.add_reward_rule(
        condition_callback=lambda old, new, action: action == 5,
        reward_callback=lambda old, new, action: -0.5
    )
    
    # Regra 2: Recompensa se houver lucro (Dinheiro atual > Dinheiro antigo)
    env.add_reward_rule(
        condition_callback=lambda old, new, action: new[2] > old[2],
        reward_callback=lambda old, new, action: float((new[2] - old[2]) * 10.0) # Bônus escalado pelo lucro
    )
    
    # Regra 3: Punição severa (Morte) se a fome (Índice 0) cair a zero
    env.add_reward_rule(
        condition_callback=lambda old, new, action: new[0] <= 0.0,
        reward_callback=lambda old, new, action: -50.0
    )
    print("Regras de recompensa dinâmicas injetadas no ambiente.")

    # 2. Pega uma observação do ambiente
    obs, info = env.reset()
    
    # 3. Prepara os dados para o PyTorch (Adiciona a dimensão de Batch = 1)
    # Na engine do jogo, você agruparia dezenas de NPCs aqui (Batching)
    tensor_npc_state = torch.tensor(obs["npc_state"]).unsqueeze(0)
    tensor_nodes = torch.tensor(obs["nodes"]).unsqueeze(0)
    
    print("\n[Input] Estado do NPC (Fome, Energia, Dinheiro, Profissao):")
    print(np.round(obs["npc_state"], 2))
    
    print("\n[Input] Assinaturas Semânticas dos Edifícios próximos (5 edifícios, 10 tokens cada):")
    print("Shape:", obs["nodes"].shape)
    
    # 4. Executa a Rede Neural (Inferência)
    with torch.no_grad(): # Desliga o cálculo de gradientes (mais rápido para inferência)
        action_logits, value = brain(tensor_npc_state, tensor_nodes)
        
        # Converte logits em probabilidades usando Softmax
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Escolhe a ação com maior probabilidade (Argmax)
        chosen_action = torch.argmax(action_probs, dim=-1).item()
    
    print("\n--- Resultados da Rede Neural ---")
    print(f"Probabilidades das Ações: {np.round(action_probs.numpy()[0], 3)}")
    print(f"Ação Escolhida (Output): {chosen_action}")
    print(f"Valor Estimado (Critic): {value.item():.3f} (Quanto a IA acha que vai lucrar)")
    
    print("\nArquitetura validada com sucesso. Pronta para integração com algoritmos de treino PPO.")