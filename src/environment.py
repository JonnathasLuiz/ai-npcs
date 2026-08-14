"""
Definição do ambiente de simulação econômica baseado no Gymnasium.
"""

from typing import Tuple, Dict, List, Callable, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import (
    MAX_NODES_PER_QUERY,
    TOKEN_VOCAB_SIZE,
    NPC_STATE_SIZE,
    ACTION_SPACE_SIZE,
)

# Tipagem para os callbacks de recompensa:
# Condition Callback recebe (old_state, new_state, action) e retorna bool
ConditionCallback = Callable[[np.ndarray, np.ndarray, int], bool]
# Reward Callback recebe (old_state, new_state, action) e retorna float
RewardCallback = Callable[[np.ndarray, np.ndarray, int], float]


class EconomicEnv(gym.Env):
    """
    Ambiente Headless (sem interface gráfica) para treinar os NPCs usando Aprendizado por Reforço.

    Focado na performance matemática pura, permitindo a execução de milhões de ticks por segundo.
    Suporta regras dinâmicas e customizáveis de recompensa por meio de injeção de callbacks.
    """

    # Variáveis com anotações de tipo explícitas
    action_space: spaces.Discrete
    observation_space: spaces.Dict
    current_step: int
    max_steps: int
    reward_callbacks: List[Tuple[ConditionCallback, RewardCallback]]
    _current_npc_state: Optional[np.ndarray]

    def __init__(self, max_steps: int = 1000) -> None:
        """
        Inicializa o ambiente de simulação econômica.

        Args:
            max_steps (int): Número máximo de passos (ticks) por episódio antes do truncamento.
        """
        super(EconomicEnv, self).__init__()

        # Define o espaço de ação discreto (Output da rede neural)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Define o espaço de observação estruturado (Input da rede neural)
        # Separamos o estado interno do NPC do mundo externo (edifícios/nós ao redor)
        self.observation_space = spaces.Dict({
            "npc_state": spaces.Box(low=0.0, high=1.0, shape=(NPC_STATE_SIZE,), dtype=np.float32),
            "nodes": spaces.Box(low=0.0, high=1.0, shape=(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE), dtype=np.float32)
        })

        self.current_step = 0
        self.max_steps = max_steps

        # Sistema de Recompensa Customizável por Callbacks
        self.reward_callbacks = []

        # Estado atual guardado na memória para calcular as diferenças de transição (deltas)
        self._current_npc_state = None

    def add_reward_rule(
        self,
        condition_callback: ConditionCallback,
        reward_callback: RewardCallback
    ) -> None:
        """
        Injeta uma regra dinâmica de recompensa no ambiente de forma similar a um sistema assert.

        Args:
            condition_callback (ConditionCallback): Função que avalia o estado atual, o próximo
                estado e a ação tomada para decidir se a recompensa deve ser aplicada.
            reward_callback (RewardCallback): Função que calcula e retorna o valor numérico
                da recompensa/punição com base na transição de estados.
        """
        self.reward_callbacks.append((condition_callback, reward_callback))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicia o estado do episódio quando o NPC morre ou o tempo máximo é atingido.

        Args:
            seed (Optional[int]): Semente aleatória opcional para reprodutibilidade.
            options (Optional[Dict[str, Any]]): Opções adicionais para a reinicialização.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Um dicionário contendo as observações
                iniciais ("npc_state" e "nodes") e um dicionário de informações adicionais.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Gera estados iniciais simulados de forma aleatória para demonstração/treino
        self._current_npc_state = np.random.rand(NPC_STATE_SIZE).astype(np.float32)
        nodes: np.ndarray = np.random.rand(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE).astype(np.float32)

        observation = {
            "npc_state": self._current_npc_state,
            "nodes": nodes
        }
        return observation, {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executa um passo (tick) de simulação com a ação tomada pela IA.

        Aqui ocorre a lógica principal do jogo (ex: se mover, trabalhar, comprar recursos, comer).
        As recompensas são avaliadas dinamicamente usando os callbacks injetados.

        Args:
            action (int): Ação escolhida pelo cérebro do NPC.

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
                - observation: O próximo estado do NPC e do ambiente ao redor.
                - reward: Recompensa acumulada calculada de acordo com as regras injetadas.
                - terminated: Booleano indicando se o episódio terminou por falecimento ou falha crítica.
                - truncated: Booleano indicando se o episódio foi interrompido por limite de tempo.
                - info: Metadados ou informações diagnósticas adicionais.
        """
        self.current_step += 1

        # --- Simulação de Transição de Estado (Mock para demonstração) ---
        # Em produção, a lógica do jogo modificaria os atributos de forma deterministicamente matemática.
        next_npc_state: np.ndarray = np.random.rand(NPC_STATE_SIZE).astype(np.float32)
        next_nodes: np.ndarray = np.random.rand(MAX_NODES_PER_QUERY, TOKEN_VOCAB_SIZE).astype(np.float32)

        # --- Avaliação Dinâmica de Recompensas ---
        reward: float = 0.0

        # Executa as regras de recompensa configuradas
        for condition_cb, reward_cb in self.reward_callbacks:
            # Garante que temos um estado inicial antes de avaliar
            old_state = self._current_npc_state if self._current_npc_state is not None else next_npc_state
            if condition_cb(old_state, next_npc_state, action):
                reward += reward_cb(old_state, next_npc_state, action)

        # Condições de encerramento do episódio (morte ou limite de passos)
        terminated: bool = False
        truncated: bool = self.current_step >= self.max_steps

        # Atualiza o estado atual com a transição de estado realizada
        self._current_npc_state = next_npc_state

        observation = {
            "npc_state": next_npc_state,
            "nodes": next_nodes
        }

        return observation, reward, terminated, truncated, {}
