# Braço VRAM × Vertentes — Estudo comparativo de modelos

Braço separado do `COST-STUDY.md`. Pergunta central: **dado um orçamento de VRAM, o que rende mais para o RecursiveMAS — modelo pequeno em precisão cheia, modelo maior quantizado, ou MoE quantizado?** E como isso escala com a VRAM.

## Constantes do estudo

- **Framework:** RecursiveMAS, estilo de referência = **Sequential** (Planner→Critic→Solver). Os outros 3 estilos entram com o mesmo tratamento (§5).
- **Teacher (data-gen):** **GLM 5.2** via API (`z-ai/glm-5.2`, `zai-org/GLM-5.2`, $1.20/$4.10 Mtok, ctx 1M) — **em todas as células**. Gera os targets por papel **uma vez** e reusa em todo o estudo (os alvos dependem das perguntas/papéis, não do modelo-base). Não roda local.
- **Eval:** mesmos benchmarks/protocolo em todas as células (§4).
- **Escopo de custo:** só **API + máquina (GPU)**. Mão de obra não é contabilizada.

## Premissas de hardware (GPU cloud, jun/2026)

| Tier | GPU típico | $/h (community) |
|---|---|---|
| 8GB | RTX 3060/4060 | ~0,15 |
| 16GB | RTX 4060Ti 16 / T4 / A4000 | ~0,30 |
| 24GB | RTX 4090 / A10 / L4 | ~0,40 |
| 32GB | RTX 5090 | ~0,60 |
| 48GB | A6000 / L40S / A40 | ~0,85 |

VRAM por param (peso + overhead): fp16 ≈ 2,0 GB/B · 8-bit ≈ 1,0 · 4-bit ≈ 0,6.

---

## 1. As vertentes (o que cada braço isola)

| Vertente | Definição | Precisão | O que isola | Exemplos (catálogo jun/2026) |
|---|---|---|---|---|
| **V1 — Pequeno simples** | maior denso que cabe **em precisão cheia** | fp16/bf16 | baseline de **hidden state limpo** | `qwen3-8b`, `qwen3-14b` |
| **V2 — Denso quantizado** | denso **maior** comprimido pro mesmo VRAM | 8/4-bit | **tamanho × ruído de quant** (denso) | `qwen3.5-27b`@8bit, `qwen3-32b`@4bit, `gemma-4-31b`@4bit |
| **V3 — MoE quantizado** | MoE **poucos-ativos** comprimido | 8/4-bit | **denso × MoE** + velocidade (ativos) | `gemma-4-26b-a4b`@8bit, `qwen3.5-35b-a3b`@4bit, `nemotron-30b-a3b`@4bit |

8-bit é tratado como sub-caso de V2/V3 (não vertente própria).

---

## 2. Matriz — estilo Sequential (papéis explícitos, 3 agentes co-residentes, +25% margem de ativação)

Papéis: **Planner** (decompõe) · **Critic** (julga/refina) · **Solver** (gera final). Sizing Planner ≥ Critic ≥ Solver.

| Tier | Vert. | **Planner** | **Critic** | **Solver** | GPU$ | Capac. |
|---|---|---|---|---|---|---|
| **8GB** | V1 | `qwen3-1.7b` | `qwen3-0.6b` | `qwen3-0.6b` | $1,5 | baixíssima |
| | V2 | `qwen3-4b`@4bit | `qwen3-4b`@4bit | `qwen3-1.7b`@4bit | $1,8 | baixa |
| | V3 | — MoE não co-fit | — | — | — | — |
| **16GB** | V1 | `qwen3-1.7b` | `qwen3-1.7b` | `qwen3-1.7b` | $4,2 | baixa |
| | V2 | `qwen3-8b`@4bit | `qwen3-8b`@4bit | `qwen3-4b`@4bit | $5,4 | média-baixa |
| | V3 | — só 1 MoE | — | — | — | — |
| **24GB** | V1 | `qwen3-4b` | `qwen3-4b` | `qwen3-1.7b` | $8,0 | média-baixa |
| | V2 | `qwen3-14b`@4bit | `qwen3-8b`@4bit | `qwen3-4b`@4bit | $11,2 | média |
| | V3 | `gemma-4-26b-a4b`@4bit | `qwen3-1.7b`@4bit | `qwen3-1.7b`@4bit | $9,6 | média-alta |
| **32GB** | V1 | `qwen3-8b` | `qwen3-4b` | `qwen3-4b` | $15,6 | média |
| | V2 | `qwen3.5-27b`@4bit | `qwen3-8b`@4bit | `qwen3-4b`@4bit | $21,6 | média-alta |
| | V3 | `gemma-4-26b-a4b`@8bit | `qwen3-1.7b`@4bit | `qwen3-1.7b`@4bit | $18,0 | **alta** (8-bit limpo) |
| **48GB** | V1 | `qwen3-8b` | `qwen3-8b` | `qwen3-4b` | $27,2 | média-alta |
| | V2 | `qwen3-32b`@4bit | `qwen3-14b`@4bit | `qwen3-8b`@4bit | $42,5 | alta |
| | V3 | `gemma-4-26b-a4b`@4bit | `nemotron-30b-a3b`@4bit | `qwen3-4b`@4bit | $34,0 | alta |
| **Teacher** | — | **GLM 5.2 (API)** — gera os 3 alvos/amostra, 1× p/ todo o estudo | | | **$24–240** | constante |

**GPU$ = $/h × GPU-h**, com GPU-h (treino dos links + eval) por config: base ~20 GPU-h em modelos pequenos, escalado por tamanho/overhead de quant:

| Tier | V1 GPU-h | V2 GPU-h | V3 GPU-h |
|---|---|---|---|
| 8GB | 10 | 12 | — |
| 16GB | 14 | 18 | — |
| 24GB | 20 | 28 | 24 |
| 32GB | 26 | 36 | 30 |
| 48GB | 32 | 50 | 40 |

**Notas de papel/VRAM:**
- **Critic reusa o base do Planner** quando é o mesmo modelo (1 carga de pesos + RecursiveLink/LoRA própria) → libera VRAM.
- **V3 (MoE) força config assimétrica:** o MoE quantizado come quase toda a VRAM (total de params residente, não só ativos) → vira **Planner MoE pesado + Critic/Solver leves** até 48GB (onde cabem 2 MoEs).

---

## 3. Custo — rollup

**Sequential (13 células viáveis):** V1 $56,5 + V2 $82,5 + V3 $61,6 = **~$200 GPU**.

**Demais estilos** (mesmo padrão de matriz, multiplicador sobre o Sequential por nº de agentes/eval):
- Mixture (4–5 agentes) ≈ ×1,4 → ~$280
- Distillation (2 agentes) ≈ ×0,6 → ~$120
- Deliberation (2 agentes) ≈ ×0,7 → ~$140 + Tavily ($0–100)

**Teacher GLM 5.2 (1×, Sequential+Distillation usam):** $24–240.

**Estudo completo (4 estilos × 5 tiers × vertentes):** **≈ $750–1.100 GPU + $24–240 teacher ≈ $800–1.300** (+ Tavily $0–100).

> O teacher (linha fixa) **domina o custo nos tiers baixos** e some nos altos.

---

## 4. Protocolo de avaliação

- **Benchmarks (subset representativo):** MATH500, GPQA-Diamond, LiveCodeBench-v6, HotpotQA (este p/ Deliberation). Corta wall-clock vs os 9 completos.
- **Profundidades:** r ∈ {1, 2, 3}.
- **Pass@10 do AIME:** fora deste braço (caro; fica no estudo-base).
- **Métricas por célula:**
  1. Acurácia nos benchmarks.
  2. tok/s + wall-clock de eval.
  3. **Fidelidade do hidden state** = Δacurácia vs a célula **V1 fp16 do mesmo tier** (V1 = referência de hidden state limpo).
  4. Custo total = GPU $/h × GPU-h + teacher constante.

**Comparações pareadas:**
- **V1 vs V2** (mesmo tier) → "vale trocar precisão limpa por mais parâmetros quantizados?" (pergunta central).
- **V2 vs V3** (mesmo tier) → "MoE compensa o custo de VRAM dos experts?" (denso × MoE).
- **Mesma vertente entre tiers** → retorno marginal da VRAM.

---

## 5. Matrizes dos outros 3 estilos (papéis por célula)

GPU$ derivado do Sequential pelo multiplicador de estilo: **Mixture ×1,4 · Distillation ×0,6 · Deliberation ×0,7**. Células inviáveis por co-residência marcadas `—`.

### 5.1 Mixture (Math ∥ Code ∥ Science/Med → Summarizer; 4 agentes co-residentes)

| Tier | V | Math | Code | Science/Med | Summarizer | GPU$ |
|---|---|---|---|---|---|---|
| **8GB** | — | inviável (4 agentes não cabem) | | | | — |
| **16GB** | V1 | `r1-distill-1.5b` | `qwen2.5-coder-3b`† | `qwen3-1.7b` | `qwen3-1.7b` | $5,9 |
| | V2 | `r1-distill-7b`@4bit | `Seed-Coder-8B`@4bit | `qwen3-4b`@4bit | `qwen3-1.7b`@4bit | $7,6 |
| | V3 | — (MoE não co-fit) | | | | — |
| **24GB** | V1 | `r1-distill-1.5b` | `Seed-Coder-8B`‡ | `medgemma-4b` | `qwen3-1.7b` | $11,2 |
| | V2 | `r1-distill-7b`@4bit | `Seed-Coder-8B`@4bit | `qwen3-8b`@4bit | `qwen3-4b`@4bit | $15,7 |
| | V3 | `r1-1.5b`@4bit | **`qwen3-coder-30b-a3b`@4bit** | `qwen3-1.7b`@4bit | `qwen3-1.7b`@4bit | $13,4 |
| **32GB** | V1 | `r1-distill-1.5b` | `Seed-Coder-8B` | `qwen3-4b` | `qwen3-1.7b` | $21,8 |
| | V2 | `r1-distill-7b`@4bit | `Seed-Coder-8B`@8bit | `qwen3-8b`@8bit | `qwen3-4b`@4bit | $30,2 |
| | V3 | `r1-1.5b`@4bit | `qwen3-coder-30b-a3b`@4bit | `gemma-4-26b-a4b`@4bit | `qwen3-1.7b`@4bit | $25,2 |
| **48GB** | V1 | `r1-distill-1.5b` | `Seed-Coder-8B` | `qwen3-8b` | `qwen3-4b` | $38,1 |
| | V2 | `r1-distill-7b`@4bit | `qwen2.5-coder-32b`@4bit | `qwen3-14b`@4bit | `qwen3-4b`@4bit | $59,5 |
| | V3 | `r1-1.5b`@4bit | `qwen3-coder-30b-a3b`@4bit | `gemma-4-26b-a4b`@4bit | `qwen3-1.7b`@4bit | $47,6 |

† `qwen2.5-coder-3b` é **licença não-comercial** — só em teste/pesquisa; senão usar coder geral pequeno. ‡ `Seed-Coder-8B` fp16 em 24GB é apertado com os outros 3 — co-residência no limite. **É o estilo ideal pro eixo denso×MoE** (Code/Science têm MoE e denso bons).

### 5.2 Distillation (Expert + Learner; 2 agentes — folgado)

**Melhor estilo para isolar V1 × V2:** Expert pequeno-cheio vs Expert grande-quantizado, com Learner controlado.

| Tier | V | Expert | Learner | GPU$ |
|---|---|---|---|---|
| **8GB** | V1 | `qwen3-1.7b` | `qwen3-0.6b` | $0,9 |
| | V2 | `qwen3-4b`@4bit | `qwen3-1.7b`@4bit | $1,1 |
| **16GB** | V1 | `qwen3-4b` | `qwen3-1.7b` | $2,5 |
| | V2 | `qwen3-8b`@4bit | `qwen3-4b`@4bit | $3,2 |
| **24GB** | V1 | `qwen3-8b` | `qwen3-1.7b` | $4,8 |
| | V2 | `qwen3-14b`@4bit | `qwen3-4b`@4bit | $6,7 |
| | V3 | `gemma-4-26b-a4b`@4bit | `qwen3-1.7b`@4bit | $5,8 |
| **32GB** | V1 | `qwen3-8b` | `qwen3-4b` | $9,4 |
| | V2 | `qwen3.5-27b`@4bit | `qwen3-4b`@4bit | $13,0 |
| | V3 | `gemma-4-26b-a4b`@8bit | `qwen3-4b`@4bit | $10,8 |
| **48GB** | V1 | `qwen3-14b` | `qwen3-4b` | $16,3 |
| | V2 | `qwen3-32b`@4bit | `qwen3-8b`@4bit | $25,5 |
| | V3 | `gemma-4-26b-a4b`@8bit | `qwen3-8b`@4bit | $20,4 |

### 5.3 Deliberation (Reflector + Tool-Caller + Tavily; 2 agentes)

| Tier | V | Reflector | Tool-Caller | GPU$ |
|---|---|---|---|---|
| **8GB** | V1 | `qwen3-1.7b` | `qwen3-1.7b` | $1,1 |
| | V2 | `qwen3-4b`@4bit | `qwen3-4b`@4bit | $1,3 |
| **16GB** | V1 | `qwen3-4b` | `qwen3-1.7b` | $2,9 |
| | V2 | `qwen3-8b`@4bit | `qwen3-8b`@4bit | $3,8 |
| **24GB** | V1 | `qwen3-4b` | `qwen3-4b` | $5,6 |
| | V2 | `qwen3-14b`@4bit | `qwen3-8b`@4bit | $7,8 |
| | V3 | `gemma-4-26b-a4b`@4bit | `qwen3-1.7b`@4bit | $6,7 |
| **32GB** | V1 | `qwen3-8b` | `qwen3-4b` | $10,9 |
| | V2 | `qwen3.5-27b`@4bit | `qwen3-8b`@4bit | $15,1 |
| | V3 | `gemma-4-26b-a4b`@8bit | `qwen3-1.7b`@4bit | $12,6 |
| **48GB** | V1 | `qwen3-8b` | `qwen3-8b` | $19,0 |
| | V2 | `qwen3-32b`@4bit | `qwen3-14b`@4bit | $29,8 |
| | V3 | `gemma-4-26b-a4b`@4bit | `nemotron-30b-a3b`@4bit | $23,8 |

Tool-Caller precisa de **tool-calling confiável** (Qwen3 atende) + `TAVILY_API_KEY`. **Tavily** no eval = custo de busca **$0–100** sobre o braço todo (não por célula).

---

## 6. Rigor de quantização + VRAM

- **Treino dos links:** bitsandbytes **NF4 (QLoRA-style)** sobre base congelada; só os ~13M params dos links treinam.
- **Eval:** **AWQ/GPTQ** W4/W8.
- **Caveat:** verificar `output_hidden_states` no caminho quantizado (HF transformers expõe; vLLM/AWQ às vezes não) — o RecursiveLink depende disso.
- **VRAM real = pesos + ativações do unroll recursivo** → ligar gradient checkpointing. **A profundidade máx de recursão cai nos tiers baixos** (8GB possivelmente só r=1–2); declarar por célula ao executar.

---

## 7. Defaults travados

- Estilos: **todos os 4** (Distillation/Mixture são os mais informativos para V1×V2 e denso×MoE).
- N de treino: **2k** (sobe pra 20k só se quiser fidelidade ao paper).
- 8-bit: sub-caso de V2/V3, não vertente nova.
- Cobertura: **todas as 5 tiers (8/16/24/32/48GB)** apresentadas; células inviáveis por co-residência marcadas `—` (§2, §5).

## 8. Hipóteses que o estudo testa

1. **V1 (hidden state limpo) pode bater V2/V3** apesar de menor — porque o RecursiveLink depende do latente, não só da "inteligência" bruta do modelo.
2. **8-bit (V2/V3 nos tiers ≥32GB) é o ponto ótimo:** capacidade alta sem ruído pesado de quantização.
3. Há um **joelho na curva VRAM×qualidade** onde gastar mais GPU para de compensar (provável 24–32GB).
