# J-NPCs: Simulador de Economia com IA (PPO Actor-Critic + Multi-Head Attention)

Este repositório contém a arquitetura de Inteligência Artificial para um simulador de economia headless focado em NPCs autônomos. Os NPCs utilizam Aprendizado por Reforço (PPO - Proximal Policy Optimization) estruturado em uma rede neural Actor-Critic combinada com um poderoso mecanismo de Atenção Multi-Head para tomada de decisão dinâmica com base no ambiente ao seu redor.

O projeto foi totalmente modularizado para melhor organização, manutenibilidade e escalabilidade, contendo também tipagem estática completa (`Type Hints`) e documentação em Português Brasileiro.

---

## 🏗️ Estrutura do Projeto

A arquitetura do software é dividida em módulos independentes localizados na pasta `src/`:

```bash
.
├── j-npcs.py            # Ponto de entrada do simulador (executa a simulação de teste)
├── README.md            # Documentação completa do projeto
└── src/
    ├── __init__.py      # Inicializador do pacote Python
    ├── config.py        # Configurações globais e constantes da simulação
    ├── environment.py   # Classe EconomicEnv (Ambiente Gymnasium customizado)
    ├── brain.py         # Classe NPCBrain (Rede Neural PyTorch)
    └── main.py          # Script principal de execução da simulação
```

---

## 🧠 Arquitetura de Inteligência Artificial

A inteligência de cada NPC é modelada por meio de uma arquitetura **Actor-Critic (PPO)** que combina o processamento das necessidades internas com informações contextuais do mundo físico ao redor usando **Atenção (Multi-Head Attention)**.

```
                    ┌─────────────────────────┐
                    │  Necessidades do NPC    │ (Fome, Energia, Dinheiro, Profissão)
                    └───────────┬─────────────┘
                                │
                                ▼
                       [ State MLP (Linear) ]
                                │
                                ▼ [Query]
                      ┌───────────────────┐
                      │    Mecanismo de   │◄── [ Node Embedding (Linear) ] ── [ Edifícios Próximos ]
                      │ Atenção Multi-Head│
                      └─────────┬─────────┘
                                │
                                ▼ [Vetor de Contexto]
                      ┌───────────────────┐
                      │ Camada de Fusão   │ (Estado Interno + Contexto do Mapa)
                      └─────────┬─────────┘
                                │
                                ├───► [ Actor Head ]  ──► Distribuição de Ações (Probabilidade)
                                │
                                └───► [ Critic Head ] ──► Valor do Estado (Retorno Esperado)
```

### 1. Processamento de Necessidades (Via 1)
O estado interno do NPC (Fome, Energia, Dinheiro e Profissão) é processado através de uma rede neural totalmente conectada (MLP) definida em `src/brain.py`. Este vetor de tamanho 4 é projetado em uma representação latente rica com dimensão de tamanho 32, servindo como a **Query (Consulta)** para o mecanismo de atenção.

### 2. Processamento Espacial / Nós Semânticos (Via 2)
O NPC enxerga até 5 edifícios próximos ao mesmo tempo. Cada edifício possui uma assinatura semântica com 10 tokens de recurso. Essas assinaturas são mapeadas para um espaço latente de dimensão 32, servindo como as **Keys (Chaves)** e **Values (Valores)** para o mecanismo de atenção.

### 3. Mecanismo de Atenção Multi-Head
Utilizando a camada `nn.MultiheadAttention`, o NPC cruza ativamente suas necessidades com as oportunidades presentes no ambiente físico.

* **Query (Q):** O que o NPC quer ou precisa no momento (ex: se está com muita fome, a consulta expressará alta necessidade de comida).
* **Keys (K) & Values (V):** O que cada edifício ao redor oferece ou representa (ex: um mercado que vende comida terá uma chave correspondente).

A atenção permite que o NPC decida, dinamicamente, em qual edifício ele deve focar sua atenção de acordo com seu estado interno, produzindo um **Vetor de Contexto** que consolida de forma inteligente o cenário atual.

### 4. Fusão e Tomada de Decisão (Actor-Critic)
O vetor de contexto é concatenado ao estado interno do NPC, resultando em um vetor de tamanho 64. Esse vetor consolidado passa por camadas densas de decisão que alimentam duas saídas (Heads):
* **Actor (Ator):** Gera logits que representam a distribuição de probabilidade das ações. A ação com maior probabilidade é escolhida pelo NPC.
* **Critic (Crítico):** Estima um valor contínuo que avalia o quão boa/lucrativa é a situação atual para o NPC (estimativa de recompensa futura).

---

## 🎯 Sistema Dinâmico de Recompensas (`reward_callbacks`)

Ao invés de programar recompensas estáticas diretamente no código do ambiente, o `EconomicEnv` (em `src/environment.py`) implementa um sistema altamente desacoplado baseado em regras injetáveis (`reward_callbacks`).

Isso funciona de forma idêntica a um sistema de asserção de agentes (`agent.assert(status, callback)`), permitindo acoplar novas dinâmicas de jogo sem alterar a classe do ambiente:

1. **Condition Callback (`condition_callback`):** Uma função que avalia as mudanças de estados (`old_state`, `new_state`, `action`) e retorna `True` ou `False`.
2. **Reward Callback (`reward_callback`):** Uma função executada apenas quando a condição é verdadeira, responsável por calcular o valor numérico da recompensa ou punição.

### Regras Padrão Injetadas no Simulador:
* **Ficar parado:** O NPC recebe uma punição leve (-0.5) se tomar a ação de ficar parado (Ação 5), incentivando-o a explorar e agir.
* **Lucro Financeiro:** O NPC recebe uma recompensa positiva escalada em 10x quando seu dinheiro atual é superior ao dinheiro que possuía no passo anterior.
* **Morte por Inanição:** Se o atributo de fome (Índice 0) cair para um valor menor ou igual a zero, o NPC sofre uma penalidade severa (-50.0).

---

## ⚙️ Instalação e Execução

### Pré-requisitos
Certifique-se de ter o Python 3.12+ instalado na sua máquina, bem como o gerenciador de pacotes `pip`.

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/ai-npcs.git
cd ai-npcs
```

### 2. Instalar as Dependências
O projeto utiliza bibliotecas fundamentais de aprendizado por reforço e aprendizado profundo:
* **PyTorch** (Processamento tensorial e redes neurais)
* **Gymnasium** (Estrutura padrão de ambientes de RL)
* **NumPy** (Computação científica e operações com vetores)

Você pode instalar as dependências necessárias executando:
```bash
pip install torch numpy gymnasium
```

### 3. Executar o Simulador
Para rodar a verificação e o fluxo de demonstração da rede executando inferência no ambiente modularizado:
```bash
python3 j-npcs.py
```

Você verá no terminal a inicialização das regras dinâmicas, o estado simulado do NPC, as assinaturas do mapa e o resultado probabilístico da tomada de decisão realizada pelo cérebro da IA.
