"""
Script principal de execução do simulador de economia com Inteligência Artificial.
Demonstra a criação do ambiente, configuração das regras de recompensa e inferência na rede neural.
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.environment import EconomicEnv
from src.brain import NPCBrain


def run_simulation() -> None:
    """
    Executa a inicialização e uma rodada de teste (inferência) da arquitetura de IA dos NPCs.
    """
    print("--- Inicializando Arquitetura de IA para Economia (Modularizada) ---")

    # 1. Instanciar o ambiente e a rede neural (cérebro)
    env = EconomicEnv()
    brain = NPCBrain()

    # 2. Configurar as regras dinâmicas e customizáveis de recompensa
    # Lembrete dos índices no array de estados do NPC:
    # 0 = Fome, 1 = Energia, 2 = Dinheiro, 3 = Profissão

    # Regra 1: Punição leve por permanecer inativo (ação 5)
    env.add_reward_rule(
        condition_callback=lambda old, new, action: action == 5,
        reward_callback=lambda old, new, action: -0.5
    )

    # Regra 2: Recompensa se houver aumento financeiro (lucro)
    env.add_reward_rule(
        condition_callback=lambda old, new, action: new[2] > old[2],
        reward_callback=lambda old, new, action: float((new[2] - old[2]) * 10.0)
    )

    # Regra 3: Punição severa por inanição extrema (Fome aproximando-se de zero ou menor)
    env.add_reward_rule(
        condition_callback=lambda old, new, action: new[0] <= 0.0,
        reward_callback=lambda old, new, action: -50.0
    )

    print("Regras de recompensa dinâmicas injetadas com sucesso no ambiente.")

    # 3. Reiniciar o ambiente e coletar a primeira observação espacial/interna
    obs, info = env.reset()

    # 4. Converter as observações do Gymnasium em Tensores PyTorch com dimensão de Batch (tamanho 1)
    tensor_npc_state = torch.tensor(obs["npc_state"]).unsqueeze(0)
    tensor_nodes = torch.tensor(obs["nodes"]).unsqueeze(0)

    print("\n[Input] Estado do NPC (Fome, Energia, Dinheiro, Profissao):")
    print(np.round(obs["npc_state"], 2))

    print("\n[Input] Assinaturas Semânticas dos Edifícios próximos (5 edifícios, 10 tokens cada):")
    print("Shape:", obs["nodes"].shape)

    # 5. Executar a inferência na rede neural sem calcular gradientes (modo de produção/teste rápido)
    with torch.no_grad():
        action_logits, value = brain(tensor_npc_state, tensor_nodes)

        # Converte logits brutos em probabilidades usando a função Softmax
        action_probs = F.softmax(action_logits, dim=-1)

        # Escolhe a ação com a maior probabilidade (Argmax)
        chosen_action: int = torch.argmax(action_probs, dim=-1).item()

    # 6. Exibir os resultados obtidos pela decisão da IA
    print("\n--- Resultados da Rede Neural ---")
    print(f"Probabilidades das Ações: {np.round(action_probs.numpy()[0], 3)}")
    print(f"Ação Escolhida (Output): {chosen_action}")
    print(f"Valor Estimado (Critic): {value.item():.3f} (Quanto a IA acha que vai lucrar)")

    print("\nArquitetura validada com sucesso. Pronta para integração com algoritmos de treino PPO.")


if __name__ == "__main__":
    run_simulation()
