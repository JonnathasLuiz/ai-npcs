# 📘 IMPLEMENTATION.md - Guia Completo de Uso

## 📑 Índice

1. [Visão Geral](#visão-geral)
2. [Usar Modelo Já Treinado (Inferência)](#usar-modelo-já-treinado-inferência)
3. [Adicionar Novos Tipos de Dados](#adicionar-novos-tipos-de-dados)
4. [Treinar o Modelo](#treinar-o-modelo)
5. [Exemplos Práticos](#exemplos-práticos)

---

## 🎯 Visão Geral

Este projeto implementa uma **rede neural Actor-Critic (PPO)** que controla NPCs em um jogo/simulação. O sistema é modular e permite:

- ✅ **Usar** um modelo já treinado para obter ações
- ✅ **Expandir** o modelo com novos tipos de observação (edifícios, inimigos, condições de mercado, etc)
- ✅ **Treinar** o modelo com algoritmo PPO

**Princípio de Abstração:** O programador só precisa passar os dados corretos. A rede neural os processa automaticamente.

---

## 🎮 Usar Modelo Já Treinado (Inferência)

### Nível 1: Uso Básico (Abstração Máxima)

Se você **só quer usar o modelo**, não precisa entender a rede neural:

```python
from src.environment import EconomicEnv
from src.brain import NPCBrain
import torch

# 1. Carregar o modelo treinado
brain = NPCBrain()
brain.load_state_dict(torch.load("modelo_treinado.pt"))
brain.eval()  # Modo de inferência

# 2. Obter dados do jogo
env = EconomicEnv()
obs, _ = env.reset()

# 3. Passar dados e obter ação
# ⚠️ Você só precisa passar obs - a conversão é automática!
action = get_npc_action(brain, obs)

# 4. Executar ação no jogo
obs, reward, done, truncated, info = env.step(action)
```

### Nível 2: Uso com Controle (Abstração Parcial)

Se você quer **entender o fluxo**, aqui está:

```python
import torch
from src.environment import EconomicEnv
from src.brain import NPCBrain

# Carregar modelo
brain = NPCBrain()
brain.load_state_dict(torch.load("modelo_treinado.pt"))
brain.eval()

# Obter observações do jogo
env = EconomicEnv()
obs, _ = env.reset()

# Converter observações em Tensores (o que o modelo espera)
tensor_npc_state = torch.tensor(obs["npc_state"]).unsqueeze(0)
tensor_nodes = torch.tensor(obs["nodes"]).unsqueeze(0)
tensor_building_data = torch.tensor(obs["building_data"]).unsqueeze(0)
tensor_enemies = torch.tensor(obs["enemies"]).unsqueeze(0)

# Inferência (obter ação)
with torch.no_grad():
    action_logits, state_value = brain(
        tensor_npc_state,      # Estado do NPC (4 valores)
        tensor_nodes,          # Tokens dos edifícios (5, 10)
        tensor_building_data,  # Dados dos edifícios (5, 4)
        tensor_enemies         # Dados dos inimigos (3, 4)
    )

# Converter logits em ação
import torch.nn.functional as F
action_probs = F.softmax(action_logits, dim=-1)
chosen_action = torch.argmax(action_probs, dim=-1).item()

# Executar ação
obs, reward, done, truncated, info = env.step(chosen_action)
```

### Nível 3: Função Auxiliar (Recomendado)

**Crie um arquivo `src/inference.py`:**

```python
"""
Módulo de Inferência - Interface simplificada para usar o modelo
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from src.brain import NPCBrain


class NPCInference:
    """Abstração para inferência - o programador não precisa se preocupar com detalhes"""
    
    def __init__(self, model_path: str = "modelo_treinado.pt"):
        """
        Args:
            model_path: Caminho para o arquivo do modelo treinado
        """
        self.brain = NPCBrain()
        self.brain.load_state_dict(torch.load(model_path))
        self.brain.eval()
    
    def get_action(self, observation: Dict[str, object]) -> int:
        """
        Obtém a ação do NPC baseado na observação
        
        Args:
            observation: Dict com chaves:
                - "npc_state": np.ndarray shape (4,)
                - "nodes": np.ndarray shape (5, 10)
                - "building_data": np.ndarray shape (5, 4)
                - "enemies": np.ndarray shape (3, 4)
        
        Returns:
            int: Ação escolhida (0-5)
        
        Exemplo de Uso:
            npc = NPCInference("modelo_treinado.pt")
            obs, _ = env.reset()
            action = npc.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
        """
        # Converter observações em Tensores (automático)
        tensor_npc_state = torch.tensor(observation["npc_state"]).unsqueeze(0)
        tensor_nodes = torch.tensor(observation["nodes"]).unsqueeze(0)
        tensor_building_data = torch.tensor(observation["building_data"]).unsqueeze(0)
        tensor_enemies = torch.tensor(observation["enemies"]).unsqueeze(0)
        
        # Inferência
        with torch.no_grad():
            action_logits, state_value = self.brain(
                tensor_npc_state,
                tensor_nodes,
                tensor_building_data,
                tensor_enemies
            )
        
        # Converter em ação
        action_probs = F.softmax(action_logits, dim=-1)
        chosen_action = torch.argmax(action_probs, dim=-1).item()
        
        return chosen_action
    
    def get_action_probs(self, observation: Dict[str, object]) -> list:
        """
        Retorna as probabilidades de cada ação (útil para análise)
        """
        tensor_npc_state = torch.tensor(observation["npc_state"]).unsqueeze(0)
        tensor_nodes = torch.tensor(observation["nodes"]).unsqueeze(0)
        tensor_building_data = torch.tensor(observation["building_data"]).unsqueeze(0)
        tensor_enemies = torch.tensor(observation["enemies"]).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _ = self.brain(
                tensor_npc_state,
                tensor_nodes,
                tensor_building_data,
                tensor_enemies
            )
        
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs[0].tolist()


# Uso:
# npc = NPCInference("modelo_treinado.pt")
# action = npc.get_action(obs)
```

**Agora basta:**

```python
from src.inference import NPCInference
from src.environment import EconomicEnv

# Inicializar
npc = NPCInference("modelo_treinado.pt")
env = EconomicEnv()

# Loop do jogo
obs, _ = env.reset()
while True:
    action = npc.get_action(obs)  # ✅ Simples!
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break
```

---

## 🔧 Adicionar Novos Tipos de Dados

### Padrão de Adição (3 Passos)

Quer adicionar um novo tipo de observação? Siga este padrão:

#### **Passo 1: Adicionar Constantes em `config.py`**

```python
# src/config.py

# Seu novo tipo de observação
MAX_ITEMS_PER_QUERY: int = 5        # Quantos itens ver simultaneamente
ITEM_FEATURES_SIZE: int = 3         # Dados por item: [tipo, raridade, distância]
```

#### **Passo 2: Adicionar ao Ambiente em `environment.py`**

```python
# src/environment.py
from src.config import (
    # ... existentes ...
    MAX_ITEMS_PER_QUERY,
    ITEM_FEATURES_SIZE,
)

class EconomicEnv(gym.Env):
    def __init__(self, max_steps: int = 1000):
        # ... código existente ...
        
        self.observation_space = spaces.Dict({
            "npc_state": spaces.Box(...),
            "nodes": spaces.Box(...),
            "building_data": spaces.Box(...),
            "enemies": spaces.Box(...),
            # NOVO: Adicionar seu tipo
            "items": spaces.Box(
                low=0.0, high=1.0,
                shape=(MAX_ITEMS_PER_QUERY, ITEM_FEATURES_SIZE),
                dtype=np.float32
            )
        })
        
        self._current_items = None
    
    def reset(self, seed=None, options=None):
        # ... código existente ...
        
        # NOVO: Gerar dados do seu tipo
        self._current_items = np.random.rand(
            MAX_ITEMS_PER_QUERY, ITEM_FEATURES_SIZE
        ).astype(np.float32)
        
        observation = {
            # ... existentes ...
            "items": self._current_items,  # NOVO
        }
        return observation, {}
    
    def step(self, action: int):
        # ... código existente ...
        
        # NOVO: Próximo estado do seu tipo
        next_items = np.random.rand(
            MAX_ITEMS_PER_QUERY, ITEM_FEATURES_SIZE
        ).astype(np.float32)
        
        # ... processar rewards ...
        
        self._current_items = next_items
        
        observation = {
            # ... existentes ...
            "items": next_items,  # NOVO
        }
        
        return observation, reward, terminated, truncated, {}
```

#### **Passo 3: Adicionar à Rede Neural em `brain.py`**

```python
# src/brain.py
from src.config import (
    # ... existentes ...
    MAX_ITEMS_PER_QUERY,
    ITEM_FEATURES_SIZE,
)

class NPCBrain(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ... camadas existentes ...
        
        # NOVO: Embedding para seu tipo
        self.item_embedding = nn.Linear(ITEM_FEATURES_SIZE, 16)
        self.item_fusion = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # NOVO: Atenção para seu tipo
        self.attention_items = nn.MultiheadAttention(
            embed_dim=16, num_heads=2, batch_first=True
        )
        
        # NOVO: Atualizar fusion_layer (mais dados = mais dimensões)
        # Antes: 80 (32 + 32 + 16)
        # Depois: 96 (32 + 32 + 16 + 16)
        self.fusion_layer = nn.Sequential(
            nn.Linear(96, 128),  # ← ATUALIZAR
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # ... resto igual ...
    
    def forward(
        self,
        npc_state: torch.Tensor,
        nodes: torch.Tensor,
        building_data: torch.Tensor,
        enemies: torch.Tensor,
        items: torch.Tensor  # NOVO
    ):
        # ... processamento existente de npc_state, nodes, building_data, enemies ...
        
        # NOVO: Processar items
        item_features = self.item_embedding(items)  # (1, 5, 16)
        item_features = self.item_fusion(item_features)  # (1, 5, 16)
        
        # NOVO: Atenção para items
        q_item = q_state.unsqueeze(1)[:, :, :16]
        attn_items, _ = self.attention_items(
            query=q_item,
            key=item_features,
            value=item_features
        )  # (1, 1, 16)
        context_items = attn_items.squeeze(1)  # (1, 16)
        
        # NOVO: Incluir na fusão final
        # Antes: cat([q_state, context_buildings, context_enemies], dim=1)
        # Depois:
        fused = torch.cat([
            q_state,                # (1, 32)
            context_buildings,      # (1, 32)
            context_enemies,        # (1, 16)
            context_items           # (1, 16) ← NOVO
        ], dim=1)  # (1, 96)
        
        features = self.fusion_layer(fused)  # (1, 64)
        
        # ... resto igual ...
```

#### **Pronto!** 🎉

Agora a rede neural processa seu novo tipo de dado automaticamente. Nenhuma outra mudança necessária!

---

### Exemplo: Adicionar "Condições de Mercado"

```python
# config.py - Adicionar:
MARKET_CONDITIONS_SIZE: int = 5  # [preço_geral, inflação, oferta, demanda, volatilidade]

# environment.py - Em reset() e step():
market_conditions = np.array([
    current_price_level,
    inflation_rate,
    supply_level,
    demand_level,
    price_volatility
]).astype(np.float32)

observation["market_conditions"] = market_conditions  # shape: (5,)

# brain.py - Adicionar:
self.market_embedding = nn.Linear(5, 16)
self.market_context = self.market_embedding(market_data)  # (1, 16)

# E incluir na fusão final
```

---

## 🏋️ Treinar o Modelo

### Nível 1: Script Simples de Treino

**Crie um arquivo `src/training.py`:**

```python
"""
Script de Treinamento com PPO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from src.environment import EconomicEnv
from src.brain import NPCBrain


class PPOTrainer:
    """Treina o modelo usando Proximal Policy Optimization (PPO)"""
    
    def __init__(self, brain: NPCBrain, lr: float = 3e-4, gamma: float = 0.99):
        """
        Args:
            brain: Instância de NPCBrain
            lr: Learning rate
            gamma: Fator de desconto (quanto valorizar recompensas futuras)
        """
        self.brain = brain
        self.optimizer = optim.Adam(brain.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = deque(maxlen=2048)
    
    def collect_experience(self, env: EconomicEnv, num_steps: int = 1000):
        """
        Coleta experiências do ambiente
        
        Args:
            env: Ambiente EconomicEnv
            num_steps: Quantos passos coletar
        """
        obs, _ = env.reset()
        
        for step in range(num_steps):
            # Converter observação em Tensor
            tensor_npc = torch.tensor(obs["npc_state"]).unsqueeze(0)
            tensor_nodes = torch.tensor(obs["nodes"]).unsqueeze(0)
            tensor_building = torch.tensor(obs["building_data"]).unsqueeze(0)
            tensor_enemies = torch.tensor(obs["enemies"]).unsqueeze(0)
            
            # Obter ação da rede neural
            with torch.no_grad():
                action_logits, state_value = self.brain(
                    tensor_npc,
                    tensor_nodes,
                    tensor_building,
                    tensor_enemies
                )
            
            # Escolher ação
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
            # Executar ação no ambiente
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Armazenar experiência (será usada no treino)
            self.memory.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'next_obs': next_obs
            })
            
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = env.reset()
    
    def train_step(self, num_epochs: int = 10):
        """
        Realiza um passo de treinamento com as experiências coletadas
        
        Args:
            num_epochs: Quantas vezes iterar sobre as experiências
        """
        experiences = list(self.memory)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for exp in experiences:
                # Converter observação em Tensor
                tensor_npc = torch.tensor(exp['obs']["npc_state"]).unsqueeze(0)
                tensor_nodes = torch.tensor(exp['obs']["nodes"]).unsqueeze(0)
                tensor_building = torch.tensor(exp['obs']["building_data"]).unsqueeze(0)
                tensor_enemies = torch.tensor(exp['obs']["enemies"]).unsqueeze(0)
                
                # Forward pass
                action_logits, state_value = self.brain(
                    tensor_npc,
                    tensor_nodes,
                    tensor_building,
                    tensor_enemies
                )
                
                # Calcular loss
                action_probs = torch.softmax(action_logits, dim=-1)
                log_prob = torch.log(action_probs[0, exp['action']])
                
                # Vantagem: recompensa realizada - valor predito
                advantage = exp['reward'] - state_value.item()
                
                # Loss do ator (política)
                actor_loss = -log_prob * advantage
                
                # Loss do crítico (valor)
                critic_loss = (advantage ** 2)
                
                # Loss total
                loss = actor_loss + 0.5 * critic_loss
                total_loss += loss.item()
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_loss = total_loss / len(experiences)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    def train_episode(self, env: EconomicEnv, num_episodes: int = 10):
        """
        Treina por múltiplos episódios
        
        Args:
            env: Ambiente
            num_episodes: Número de episódios
        """
        for episode in range(num_episodes):
            print(f"\n=== Episódio {episode+1}/{num_episodes} ===")
            
            # Coletar experiências
            self.collect_experience(env, num_steps=1000)
            
            # Treinar com as experiências
            self.train_step(num_epochs=5)
            
            # Salvar modelo a cada 5 episódios
            if (episode + 1) % 5 == 0:
                torch.save(self.brain.state_dict(), f"modelo_ep{episode+1}.pt")
                print(f"Modelo salvo: modelo_ep{episode+1}.pt")
```

**Use assim:**

```python
# train_main.py
from src.environment import EconomicEnv
from src.brain import NPCBrain
from src.training import PPOTrainer

# Criar ambiente e modelo
env = EconomicEnv()
brain = NPCBrain()

# Treinar
trainer = PPOTrainer(brain, lr=3e-4)
trainer.train_episode(env, num_episodes=100)

# Salvar modelo final
torch.save(brain.state_dict(), "modelo_final.pt")
```

---

## 💡 Exemplos Práticos

### Exemplo 1: Usar Modelo Treinado em Jogo

```python
# game_with_ai.py
from src.inference import NPCInference
from src.environment import EconomicEnv

# Carregar modelo treinado
npc_brain = NPCInference("modelo_final.pt")
env = EconomicEnv()

# Simular jogo
obs, _ = env.reset()
total_reward = 0

for step in range(1000):
    # ✅ Uma linha para obter ação!
    action = npc_brain.get_action(obs)
    
    # Executar ação
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    if done or truncated:
        break

print(f"Recompensa Total: {total_reward}")
```

### Exemplo 2: Adicionar Nova Observação (Clima)

```python
# 1. config.py
WEATHER_SIZE: int = 3  # [temperatura, precipitação, vento]

# 2. environment.py - Em reset() e step():
weather = np.array([
    temperature,
    precipitation,
    wind_speed
]).astype(np.float32)
observation["weather"] = weather

# 3. brain.py
self.weather_embedding = nn.Linear(3, 16)
weather_context = self.weather_embedding(weather_tensor)
# ... incluir na fusão final

# Pronto! ✅ Modelo agora usa clima!
```

### Exemplo 3: Análise de Comportamento

```python
# analyze_npc.py
from src.inference import NPCInference
from src.environment import EconomicEnv

npc = NPCInference("modelo_final.pt")
env = EconomicEnv()

obs, _ = env.reset()

# Ver probabilidades de ações
action_probs = npc.get_action_probs(obs)

print("Probabilidades das Ações:")
actions = [
    "Ir para Edifício 0",
    "Ir para Edifício 1",
    "Ir para Edifício 2",
    "Ir para Edifício 3",
    "Ir para Edifício 4",
    "Ficar Parado"
]

for i, prob in enumerate(action_probs):
    print(f"{actions[i]:<25}: {prob:.2%}")
```

---

## 🎓 Princípio de Abstração

A arquitetura segue este princípio:

```
PROGRAMADOR
    ↓
┌─────────────────────────────────────────┐
│ NPCInference.get_action(obs)            │ ← Interface Simples
│ Sem se preocupar com internals          │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ environment.py                           │ ← Gera Dados
│ obs = {"npc_state", "nodes", ...}       │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ brain.py                                │ ← Processa Dados
│ Conversão automática de tipos           │
│ Embedding → Atenção → Fusão            │
└─────────────────────────────────────────┘
```

**O programador só precisa:**
1. ✅ Passar `obs` correto
2. ✅ Configurar constantes em `config.py` para novos tipos
3. ✅ Seguir o padrão de 3 passos para adicionar dados

**O resto é automático!**

---

## 📚 Resumo Rápido

| Tarefa | Como Fazer |
|--------|-----------|
| **Usar modelo** | `npc = NPCInference("modelo.pt")` → `action = npc.get_action(obs)` |
| **Adicionar novo dado** | 3 passos: `config.py` + `environment.py` + `brain.py` |
| **Treinar** | `trainer = PPOTrainer(brain)` → `trainer.train_episode(env, 100)` |
| **Analisar ações** | `probs = npc.get_action_probs(obs)` |
| **Salvar modelo** | `torch.save(brain.state_dict(), "modelo.pt")` |
| **Carregar modelo** | `brain.load_state_dict(torch.load("modelo.pt"))` |

---

## ✅ Checklist de Uso

- [ ] Treinei o modelo (`PPOTrainer`)
- [ ] Salvei o modelo (`torch.save()`)
- [ ] Carrego o modelo (`NPCInference`)
- [ ] Obtenho observação do ambiente
- [ ] Passo para `get_action(obs)`
- [ ] Obtenho ação e executo no jogo

**Tudo pronto!** 🎉

