# Estudo de Custos — Recriação do RecursiveMAS com modelos recentes (mid-2026)

**Objetivo:** estimar o custo de recriar o estudo do paper RecursiveMAS (arXiv:2604.25917) trocando os modelos-base originais por modelos open-weight mais recentes.

**Escopo deste estudo:** **somente custo de API e de máquina (GPU)**. Mão de obra / engenharia **não é contabilizada**. O porte de código é tratado como pré-requisito técnico (§5 Riscos), não como item de orçamento.

**Premissa-mestre declarada:** o tamanho/família dos modelos é uma **alavanca parametrizada** (a definir). Mostro como o custo se move com essa escolha e termino pedindo as travas para fechar um número exato.

> **Atualizações (jun/2026):** o braço de comparação **VRAM × precisão** está em **`COST-STUDY-vram-tiers.md`** (5 tiers × V1 pequeno-fp16 / V2 denso-quantizado / V3 MoE-quantizado). O **teacher da recriação é GLM 5.2** (`z-ai/glm-5.2`, $1.20/$4.10 Mtok → ~$24–240 one-time, gerado 1×). Catálogo de modelos atual por papel em **`MODELS.md`**. Preços de GPU community jun/2026: 24GB ~$0,40/h · 32GB (RTX 5090) ~$0,60/h · 48GB ~$0,85/h.

---

## 0. Insight central (leia isto primeiro)

**O custo é todo GPU + API.** Em escala original fica em **dezenas a poucas centenas de dólares**; só vai a milhares se você subir para modelos frontier ou reproduzir todos os baselines.

- O treino dos `RecursiveLink` é trivial em compute: o paper reporta **US$4.27 por config** (Table 5, "scaled sequential-style"), com pico de **15.29 GB** de VRAM e só **13.12M params treináveis (0.31%)**. As LLMs-base ficam **congeladas**. É mais barato que LoRA (US$6.64) e Full SFT (US$9.67) no mesmo setup.
- **Restrição técnica que define a estrutura do custo:** o RecursiveMAS precisa do **hidden state da última camada** dos modelos. Isso exige **open-weight carregado localmente** (transformers/vLLM, `device="cuda"`, `trust_remote_code=True`). Modelos **só-API (GPT/Claude/etc.) não servem como agentes-base** — não dão acesso ao hidden state. Logo, o custo dos agentes é **aluguel de GPU**, não tokens de API. API entra em apenas **dois pontos opcionais**: (i) o **teacher de data-gen** (se você optar por gerar os targets via API em vez de self-host) e (ii) a **busca Tavily** na Deliberation.
- A VPS local (2 vCPU, 7.8 GB RAM, 8.5 GB livre, **sem GPU**) só serve para orquestrar e editar código. Todo treino/eval real roda em **GPU cloud** (Runpod/Vast/Lambda/etc.).
- Dentro da infra, os itens que pesam: **(C3) treino** quando em escala frontier, **(C4) eval** (×3 profundidades + Pass@10 no AIME + baselines), e **(C2) data-gen** com teacher grande.

### TL;DR de custo por cenário (infra pura)

| Cenário | Compute GPU (C3+C4+C5) | API (teacher data-gen + Tavily) | **Total infra** |
|---|---|---|---|
| **A — PoC mínima** (1 estilo, 4–9B, sem baselines, eval reduzido) | US\$20–95 | US\$15–60 | **US\$35–155** |
| **B — Reprodução fiel** (4 estilos, escala original, eval completo + baselines principais) | US\$230–960 | US\$60–350 | **US\$290–1.310** |
| **C — Frontier** (4 estilos, MoE maiores, 80GB, eval completo + todos baselines) | US\$1.430–6.100 | US\$200–900 | **US\$1.630–7.000** |

> Em escala original (A/B) o teto fica em ~US\$1,3k. O salto para US\$7k no Cenário C vem de **GPU 80GB/multi-GPU** para segurar modelos grandes + ativações do unroll, somado a um **teacher maior**. O item de **API** que mais pesa é o teacher de data-gen *se você o servir por API*; a Tavily é marginal.

---

## 1. Decomposição de custos

### Premissas de preço de GPU cloud (faixas — verificar no momento da execução, mid-2026)

Declaradas como faixas porque preço spot/community varia dia a dia.

| GPU | VRAM | US\$/h (faixa) | US\$/h (uso na conta) |
|---|---|---|---|
| H100 | 80 GB | 2–3 (community ~2) | **2,50** |
| A100 | 80 GB | 1,2–1,8 | **1,50** |
| A100 40GB / L40S | 40–48 GB | 0,7–1,2 | **0,95** |
| RTX 4090 (community) | 24 GB | 0,3–0,5 | **0,40** |
| A10 / L4 | 24 GB | 0,4–0,75 | **0,55** |

Uso o valor "na conta" como ponto central; os totais carregam a faixa.

---

### (C1) Engenharia de porte — FORA DE ESCOPO

Recriar com modelos novos exige portar o `RecursiveLink` (hook do hidden state + matriz de input-embedding) para cada arquitetura nova e **retreinar os Outerlinks do zero** (os liberados são específicos dos pares originais). **Isso é trabalho técnico, mas não é custo de máquina nem de API — não é contabilizado aqui.** Fica só o registro de que é pré-requisito; o risco/tempo disso aparece em §5, não no orçamento. O custo de GPU do retreino dos Outerlinks já está em **C3**.

---

### (C2) Curadoria + data-gen (custo de API ou de GPU)

**O que é:** os targets por papel são reescritos a partir de pares Q-A de **s1K (~1K), m1K (~1K), OpenCodeReasoning, ARPO-SFT**. O passo caro é o **teacher gigante**:

- **Sequential:** teacher **Qwen3.5-397B-A17B** (MoE ~397B total / ~17B ativos) reescreve cada resposta em: plano inicial (target do Planner) + plano refinado guiado por crítica (target do Critic); resposta original → target do Solver. **3 gerações por amostra.**
- **Distillation:** Expert grande gera respostas-guidance → target do Expert. **~1 geração por amostra.**
- **Mixture / Deliberation:** self-supervision e ground-truth — **sem teacher gigante** (data-gen barato).

**Premissa de volume:** N amostras a reescrever.
- **N_baixo = ~2.000** (só s1K+m1K, núcleo)
- **N_alto = ~20.000** (incluindo fatias de OpenCodeReasoning + ARPO-SFT)

#### Caminho A — Teacher via API

Deixo **N** e **p** (US\$/M-tokens de output) como variáveis.

- Tokens de output por geração ≈ **800 tok** (premissa).
- Sequential = **3 gerações/amostra** → **2.400 tok out/amostra**.

**Fórmula:** `custo ≈ N × 2.400 tok × (p / 1.000.000)`.

Exemplo numérico, **p = US\$3/M tok** (output, faixa típica de MoE grande hospedado por terceiros):

| N | tok out total | custo @ \$3/M |
|---|---|---|
| 2.000 | 4,8M | **US\$14,4** |
| 20.000 | 48M | **US\$144** |

Com input contado (~1.800 tok in/amostra @ ~US\$1/M): +US\$3,6 (N=2k) / +US\$36 (N=20k). **Faixa prática API: US\$15–200.** Se o teacher escolhido custar US\$10–15/M out (frontier caro): **US\$60–800**.

#### Caminho B — Teacher self-host (vira custo de GPU, não de API)

Um MoE de ~400B precisa de **multi-GPU 80GB**. Premissa: **8×H100 80GB**.

- Throughput offline: **~30–60 amostras/min** num nó 8×H100 (geração ~800 tok, batched).
- N=20.000 × 3 gerações = 60.000 gerações → @ 45 ger/min ≈ **22 h** de nó.
- Nó 8×H100 @ US\$2,50/h/GPU × 8 = **US\$20/h** → 22 h × US\$20 = **US\$440**.
- N=2.000 (6.000 gerações) ≈ **2,2 h** → **US\$44**. + spin-up/baixar pesos (~800GB–1,5TB): **1–3 h** ≈ US\$20–60.

**Faixa self-host: US\$60–550.** Pior que API para N pequeno; competitivo só em N grande ou se já houver nó alugado para treino.

**Resumo C2:** PoC (só Sequential, N baixo, API) ≈ **US\$15–60**. Reprodução (Sequential + Distillation, N médio) ≈ **US\$60–250**. Frontier (N alto, teacher maior) ≈ **US\$200–800**.

---

### (C3) Treino dos RecursiveLinks em GPU

**Âncora:** US\$4.27/config no paper, pico **15.29 GB** VRAM, para modelos de **~4–9B**. Cabe numa **única GPU 24GB** (4090/A10/L4) → custo de hardware muito baixo.

**Premissa de tempo:** US\$4,27 @ ~US\$0,40/h (4090 community) ⇒ **~10,7 GPU-h por config** (uso como tempo de referência para escala original).

**Multiplicadores:**
- **Nº de estilos:** 4 (Sequential, Mixture, Distillation, Deliberation).
- **Variantes:** light + scaled = ×2.
- **Sweeps:** inner-loop (cosine warm-start) + outer-loop (CE no unroll por n rounds). Premissa: **×2–4** runs por config (o unroll recursivo é sensível a lr/rounds).

**Aritmética por escala de modelo:**

*Escala original (4–9B, GPU 24GB @ ~US\$0,40/h):*
- 1 config = ~10,7 GPU-h × US\$0,40 = **US\$4,3** (bate com o paper).
- 4 estilos × 2 variantes = 8 configs → ×(2–4 sweep) = **16–32 runs** → **US\$69–138**.

*Modelos MAIORES (frontier / ~30B+ denso ou MoE):*
- Precisa segurar **base congelada + ativações do unroll recursivo** na VRAM → **80GB** (H100/A100 80GB), às vezes multi-GPU.
- Premissa: tempo por config ~3–6× maior e preço/h US\$0,40 → US\$1,50–2,50.
- 1 config ≈ 30–60 GPU-h × US\$2,00 = **US\$60–120** → 16–32 runs ≈ **US\$1.440–2.880** (pode dobrar se multi-GPU).

**Faixa C3:** PoC (1 estilo, 1–2 variantes, escala original) = **US\$10–35**. Reprodução (4 estilos, escala original) = **US\$70–300**. Frontier (4 estilos, modelos grandes, 80GB) = **US\$1.000–4.000**.

---

### (C4) Avaliação em GPU (o "sleeper")

**Matriz:** 9 benchmarks (MATH500, AIME2025 30q, AIME2026 30q, GPQA-Diamond, MedQA, LiveCodeBench-v6, MBPP Plus, HotpotQA, Bamboogle) × **3 profundidades** (r=1,2,3). **AIME usa Pass@10** (×10 amostras/questão).

**Premissa de runtime:** o paper mostra runtimes na casa de **centenas a milhares de segundos** por (benchmark × método × profundidade). Uso **ponto central de 1.200 s (20 min) por (benchmark × profundidade)** para a config principal, em GPU 24–48GB.

*Só o RecursiveMAS:*
- Base ≈ 9 benchmarks × 3 profundidades = **27 células** × 20 min = **9 h** por estilo "completo".
- AIME (2 benchmarks) com Pass@10 → AIME = 2 bench × 3 prof × 20 min × 10 = **20 h** sozinho.
- Teto por passada completa ≈ **29 GPU-h**; realista (estilos só rodam seus domínios) ≈ **12–20 GPU-h**.
- @ US\$0,55/h (A10/L4) → **US\$7–16**. @ US\$1,50 (A100 80GB) → **US\$18–44**.

*+ Todos os baselines* (Single Advanced Agents com Full SFT e LoRA, LoopLM, Recursive-TextMAS, TextGrad, Mixture-of-Agents — **6 famílias**):
- Multiplicador ≈ **×6–7** sobre o eval do RecursiveMAS.
- 12–20 GPU-h × 6,5 = **78–130 GPU-h** → @ US\$0,55–1,50 = **US\$43–195**. Com reruns/instabilidade, **US\$100–400**.

**Faixa C4:** PoC (1 estilo, eval reduzido, sem AIME Pass@10 completo, sem baselines) = **US\$5–30**. Reprodução (4 estilos, 9 bench, + baselines principais) = **US\$150–600**. Frontier (modelos grandes → preço/h maior + mais GPU-h) = **US\$400–2.000**.

> C4 é o custo "dorminhoco": cada profundidade extra e cada baseline reproduzido **multiplica** GPU-horas.

---

### (C5) Storage / egress

- Checkpoints: ~10 modelos de **1.5B–9B** + outerlinks (1.5B≈3GB, 3B≈6GB, 4B≈8GB, 7B≈14GB, 9B≈18GB fp16) ≈ **80–150 GB**.
- Teacher grande (se self-host): ~400B ≈ **800GB–1,5TB**.
- Scratch (ativações/logs/eval outputs): +50–100 GB.

**Custo:** storage em cloud de GPU é US\$0,05–0,20/GB/mês ou incluso no aluguel; egress raramente cobrado em Runpod/Vast por download de HF. 
- Sem teacher self-host: **US\$5–30**.
- Com teacher self-host (1,5TB): **US\$30–100**.

**Faixa C5:** **US\$5–100.**

---

### (C6) API de busca (Tavily, só Deliberation)

- **Tavily** (Deliberation eval + search QA: HotpotQA, Bamboogle): busca por questão × 3 profundidades × múltiplas queries.
  - Premissa: ~200–600 questões × 3 prof × ~2 buscas = **1.200–3.600 buscas**.
  - Free tier ~1.000/mês + planos baixos. **US\$0–50** (cabe quase tudo no free/low tier; com baselines de search, dobra → US\$0–100).

**Faixa C6:** **US\$0–100.**

---

### Tabela-resumo da decomposição (infra)

| Comp. | O que | PoC (A) | Reprodução (B) | Frontier (C) |
|---|---|---|---|---|
| C2 | Data-gen (teacher: API ou GPU) | US\$15–60 | US\$60–250 | US\$200–800 |
| C3 | Treino dos links (GPU) | US\$10–35 | US\$70–300 | US\$1.000–4.000 |
| C4 | Eval (GPU) | US\$5–30 | US\$150–600 | US\$400–2.000 |
| C5 | Storage/egress | US\$5–30 | US\$10–60 | US\$30–100 |
| C6 | Tavily/busca | US\$0 | US\$0–100 | US\$0–100 |
| | **Total infra** | **US\$35–155** | **US\$290–1.310** | **US\$1.630–7.000** |

> (C1 = porte de engenharia: fora de escopo, não contabilizado.)

---

## 2. Mapeamento de modelos: original → equivalente recente (mid-2026)

**A DEFINIR — você escolhe.** Onde não tenho certeza de um nome exato de versão, descrevo **por classe** em vez de inventar. Candidatos plausíveis open-weight: famílias **Qwen3, Llama 4, Gemma 3, DeepSeek-V3/R1, Mistral**.

| Papel | Original (paper) | Equivalente recente (classe) | Justificativa |
|---|---|---|---|
| **Teacher de data-gen** | Qwen3.5-397B-A17B (MoE ~397B/17B) | **MoE frontier ~200–400B recente** (DeepSeek-V3/R1; Qwen3 MoE topo; Llama 4 Maverick) | Só faz data-gen (texto) → pode ser API; não entra no forward recursivo. |
| **Sequential Planner** | Qwen3.5-9B | **denso ~7–9B recente** | Capacidade média; mesma família do Solver simplifica o outer link. |
| **Sequential Critic** | (par Qwen3.5-9B/4B) | **denso ~4–9B recente, mesma família** | Reduz mismatch de hidden/tokenizer. |
| **Sequential Solver** | Qwen3.5-4B | **denso ~4B recente** | Ganho vem da recursão, não do tamanho. |
| **Mixture-Math** | DeepSeek-R1-Distill-Qwen-1.5B | **distill de raciocínio ~1.5–3B recente** | Especialista de math pequeno. |
| **Mixture-Code** | Qwen2.5-Coder-3B | **coder ~3–7B recente** (Qwen3-Coder / Codestral-class) | Modelo coder-específico. |
| **Mixture-Science** | BioMistral-7B | **denso ~7B com viés bio/ciência** (Mistral-class / Gemma 3 ~9–12B) | BioMistral é nicho; equivalente denso. |
| **Mixture-Summarizer** | Qwen3.5-2B | **denso ~2–4B recente** | Só consolida saídas dos especialistas. |
| **Distillation Expert** | Expert-Qwen3.5-9B | **denso ~9–12B recente** | Gera guidance; ~9–12B basta. |
| **Distillation Learner** | Learner-Qwen3.5-4B | **denso ~4B recente** | Supervisionado pelo ground-truth. |
| **Deliberation Reflector** | Qwen3.5-4B | **denso ~4B recente** | Reflexão iterativa; pequeno serve. |
| **Deliberation Tool-Caller** | Qwen3.5-4B (+ Tavily) | **denso ~4B recente com bom tool-use** | Chama busca externa; precisa de tool-calling confiável + `TAVILY_API_KEY`. |

**Regras ao travar os modelos:**
1. **Mesma família dentro de um estilo** aproxima hidden sizes/tokenizers e barateia o porte.
2. **Evitar MoE nos papéis-base** se quiser baratear infra — MoE complica extração de hidden state e tende a exigir 80GB. Reservar MoE só para o **teacher** (que é texto/API).
3. Subir de escala (frontier denso/MoE nos papéis-base) joga o treino para **80GB** e multiplica C3/C4.

---

## 3. Análise de sensibilidade (infra)

| Alavanca | Efeito no custo |
|---|---|
| **(a) Tamanho dos modelos-base** | Maior driver de C3/C4. 4–9B cabe em 24GB (US\$0,40/h). ~30B+ ou MoE → 80GB (US\$1,50–2,50/h) **e** mais GPU-h pelas ativações do unroll. Salto típico: **3–8×** em C3+C4. |
| **(b) Profundidade de recursão** | Treino: mais rounds = mais ativações + tempo. Eval: r∈{1,2,3} já é ×3; r=4+ é linear-a-superlinear em GPU-h. |
| **(c) Teacher self-host vs API** | N pequeno (≤2k): **API ganha** (US\$15–60 vs ~US\$60–110 self-host). N grande (≥20k) ou nó já alugado: **self-host compete**. Para PoC, **sempre API**. |
| **(d) Reproduzir baselines** | **×6–7** em C4. Diferença entre "validar que funciona" (barato) e "reproduzir as comparações do paper" (caro). Pular baselines corta C4 em ~85%. |
| **(e) 1 estilo vs 4 estilos** | 4 estilos = ~4× a superfície de data-gen/treino/eval. 1 estilo (Sequential) é o caminho mínimo. |

**Regra de bolso:** o custo de infra é dominado por `(tamanho do modelo) × (reproduzir baselines? ×6,5 : ×1) × (nº de estilos)`.

---

## 4. Três cenários (infra)

### Cenário A — Prova de conceito mínima
**Escopo:** 1 estilo (**Sequential**), modelos recentes na escala original (~4–9B), **sem baselines**, eval reduzido (subset, r∈{1,2}), teacher **via API**, N≈2.000.

| Item | Estimativa |
|---|---|
| C2 Data-gen (API, N=2k) | US\$15–60 |
| C3 Treino (1 estilo, 24GB) | US\$10–35 |
| C4 Eval (reduzido, sem baselines) | US\$5–30 |
| C5 Storage | US\$5–30 |
| C6 Tavily | US\$0 (Sequential não usa busca) |
| **Total infra** | **US\$35–155** |

**Entrega:** valida o pipeline ponta-a-ponta (treino inner-outer + eval) num estilo, com modelos recentes. Não reproduz números do paper.

---

### Cenário B — Reprodução fiel
**Escopo:** **4 estilos**, modelos recentes de **escala equivalente à original**, eval completo nos **9 benchmarks** com r∈{1,2,3} e **AIME Pass@10**, + **baselines principais** (subconjunto: Single Advanced Agents, LoopLM, MoA). Teacher API ou self-host conforme N. Tavily na Deliberation.

| Item | Estimativa |
|---|---|
| C2 Data-gen (Sequential+Distillation, N médio) | US\$60–250 |
| C3 Treino (4 estilos × variantes × sweep) | US\$70–300 |
| C4 Eval (9 bench × 3 prof + AIME Pass@10 + baselines parciais) | US\$150–600 |
| C5 Storage | US\$10–60 |
| C6 Tavily | US\$0–100 |
| **Total infra** | **US\$290–1.310** |

**Entrega:** reprodução defensável dos resultados do paper com modelos atuais. Tabelas comparáveis.

---

### Cenário C — Ambicioso / frontier
**Escopo:** **4 estilos** com modelos recentes **MAIORES** (papéis-base denso ~30B+ ou MoE; teacher MoE grande), eval completo + **todos os baselines**.

| Item | Estimativa |
|---|---|
| C2 Data-gen (N alto, teacher maior; API caro ou self-host) | US\$200–800 |
| C3 Treino (modelos grandes → **80GB**, ativações do unroll, possível multi-GPU) | US\$1.000–4.000 |
| C4 Eval (modelos grandes, preço/h maior, todos os baselines) | US\$400–2.000 |
| C5 Storage (teacher 1,5TB + checkpoints grandes) | US\$30–100 |
| C6 Tavily | US\$0–100 |
| **Total infra** | **US\$1.630–7.000** |

**Entrega:** RecursiveMAS em escala frontier. **O salto de custo vem de (1) GPU 80GB/multi-GPU e (2) o teacher grande.**

---

## 5. Riscos e incógnitas técnicas (podem estourar a faixa)

1. **Porte de arquitetura por família, esp. MoE** — extração de hidden state em DeepSeek-V3/Llama 4/Qwen3-MoE pode exigir patch no modeling do modelo. (Risco técnico, não orçado — mas pode forçar uso de 80GB e empurrar C3.)
2. **Instabilidade do treino do unroll recursivo** — gradiente retropropaga por todo o traço; sensível a lr/rounds; pode exigir mais sweeps (estoura C3).
3. **Mismatch de hidden size / tokenizer no outer link** — composições heterogêneas exigem projeção e alinhamento; quanto mais heterogêneo, mais frágil.
4. **Outerlinks têm que ser treinados do zero** com modelos novos (os liberados são específicos dos pares originais). Já embutido em C3, mas fácil esquecer.
5. **Disponibilidade dos datasets** (s1K, m1K, OpenCodeReasoning, ARPO-SFT) — se algum mudou de formato, custa curadoria extra (C2).
6. **Custo real de eval > estimado** — se os runtimes ficarem no topo da faixa (milhares de s) e/ou reruns por flakiness, **C4 pode dobrar**. Item mais propenso a surpresa para cima.
7. **Pass@10 do AIME** multiplica amostragem nessas células — fácil subdimensionar.
8. **Disco da VPS (8.5 GB livre)** — nada de pesos toca a VPS; ela só orquestra. Tudo no cloud de GPU.

---

## 6. Recomendação final

- **A infra é barata em escala original.** Cenário A = **US\$35–155**; Cenário B = **US\$290–1.310**. Os milhares só aparecem em C (modelos grandes + 80GB) ou ao reproduzir todos os baselines (C4 ×6–7).
- **Comece pelo Cenário A** para validar o pipeline com baixo compromisso de infra antes de escalar.
- **Teacher: use API até N ~5–10k amostras.** Self-host de ~400B só compensa em volume alto ou com nó já alugado.
- **Evite MoE nos papéis-base** (a menos que o alvo seja explicitamente frontier) — é o que empurra para 80GB e infla C3/C4.
- **Não existe atalho "barato via API" para os agentes-base** — eles têm que rodar localmente em GPU (dependência de hidden state). O custo é, por natureza, **aluguel de GPU**.

### Próximo passo concreto (para fechar um número exato)

Trave três coisas:

1. **Os modelos exatos por papel** (coluna ajustável da §2) — fixa C3 e C4 (escala/VRAM/preço-h).
2. **O cenário-alvo** (A, B ou C) — fixa nº de estilos, escopo de eval e se reproduz baselines.
3. **Teacher self-host vs API** — fixa C2 e C5.

Com os três travados, fecho **um número único** (não faixa) de infra.

---

*Premissas de preço de GPU e datas de modelo são de mid-2026 e devem ser reverificadas no momento da execução. Todas as estimativas trazem a aritmética acima; onde dei ponto, é ponto central de uma faixa declarada.*
