# Catálogo de modelos por papel — atual (jun/2026)

Modelos open-weight confirmados no catálogo OpenRouter / HuggingFace em jun/2026, mapeados às vertentes do braço `COST-STUDY-vram-tiers.md`.

## Método / restrições

- RecursiveMAS troca **hidden states** via RecursiveLink → todo agente-base é **open-weight rodado local** (HF/transformers, CUDA). Modelos só-API **não servem como base** (não expõem hidden state).
- **Exceção — teacher de data-gen:** gera texto offline, não entra no forward recursivo → pode ser **API**. Teacher do estudo = **GLM 5.2**.
- OpenRouter é usado como **catálogo + referência de preço** (preço só importa pro teacher).
- Vertentes: **V1** denso fp16 (hidden limpo) · **V2** denso quantizado · **V3** MoE quantizado.

> Correção vs versão anterior (defasada): **Qwen3.5/3.6 existem** (não "Qwen3.5 fictício"); **Gemma 4** lançado (abr/26); BioMistral está datado (ctx 2K) → substituído por MedGemma/Meerkat.

---

## Teacher (data-gen, API, constante)

| Modelo | id | total/ativos | $/Mtok (in/out) | licença | nota |
|---|---|---|---|---|---|
| **GLM 5.2** ★ | `z-ai/glm-5.2` | (MoE) | **$1.20 / $4.10** | open-weight (zai-org) | **escolhido**; ctx 1M; ~$24–240 one-time |
| Qwen3.5-397B-A17B | `qwen/qwen3.5-397b-a17b` | 397B/a17B | $0.39 / $2.45 | Apache-2.0 | teacher exato do paper |
| Qwen3-235B-A22B-2507 | `qwen/qwen3-235b-a22b-2507` | 235B/a22B | $0.09 / $0.10 | Apache-2.0 | mais barato (~$1–8) |

API só vale pro teacher. Qualquer outro papel = self-host local.

---

## Sequential

| Papel | V1 (denso fp16) | V2 (denso quant) | V3 (MoE quant) |
|---|---|---|---|
| Planner | `qwen3-8b`, `qwen3-14b`, `granite-4.1-8b`, `ministral-8b-2512` | `qwen3.5-27b`@8/4bit, `qwen3-32b`@4bit, `gemma-4-31b`@4bit | `gemma-4-26b-a4b`@8bit, `qwen3.5-35b-a3b`@4bit |
| Critic | mesma família do Planner (reusa base) | idem | idem |
| Solver | `qwen3-4b`, `qwen3-1.7b` | `qwen3-8b`@4bit | — |

## Mixture

| Papel | V1 | V2/V3 |
|---|---|---|
| Math | `deepseek-r1-distill-qwen-1.5b`, `qwen3-1.7b` | `deepseek-r1-distill-qwen-7b`@4bit |
| Code | **`ByteDance-Seed/Seed-Coder-8B-Instruct`** (MIT), `qwen2.5-coder-7b` (Apache) | **`qwen3-coder-30b-a3b`@4bit** (V3) |
| Science/Med | `google/medgemma-4b-it` (ctx 128k), `dmis-lab/meerkat-7b` (MedQA 70.6) | `medgemma-27b`@4bit, `gemma-4-26b-a4b`@8bit |
| Summarizer | `qwen3-1.7b`, `qwen3-4b` | `qwen3-8b`@4bit |

> Evitar `qwen2.5-coder-3b` (licença não-comercial). `Seed-Coder-8B` = melhor coder pequeno atual (MIT, HumanEval 84.8).

## Distillation

| Papel | V1 | V2 |
|---|---|---|
| Expert | `qwen3-8b` | `qwen3-14b`/`qwen3-32b`@4bit, `qwen3.5-27b`@8bit |
| Learner | `qwen3-4b` (fixo) | — |

## Deliberation (+ Tavily)

| Papel | V1 | V2 |
|---|---|---|
| Reflector | `qwen3-4b` → `qwen3-8b` | `qwen3-14b`@4bit |
| Tool-Caller | `qwen3-4b` (bom tool-calling) | `qwen3-8b`@4bit |

---

## Referência — densos pequenos/médios atuais (24–32GB, jun/2026)

| Modelo | id | params | lic. | data |
|---|---|---|---|---|
| Qwen3 | `qwen/qwen3-8b` · `-14b` · `-32b` | 8/14/32B | Apache-2.0 | 2025-04 |
| Qwen3.5 | `qwen/qwen3.5-9b` · `-27b` | 9/27B | Apache-2.0 | 2026-02/03 |
| Gemma 3 | `google/gemma-3-4b/12b/27b-it` | 4/12/27B | Gemma (gated) | 2025-03 |
| Gemma 4 | `google/gemma-4-31b-it` · `-26b-a4b-it` | 31B · 26B/a4B | Gemma (gated) | 2026-04 |
| Mistral | `mistralai/ministral-8b/14b-2512` · `mistral-small-3.2-24b` | 8/14/24B | Apache-2.0 | 2025-12/06 |
| Nemotron 3 | `nvidia/nemotron-3-nano-30b-a3b` | 30B/a3B | open | 2025-12 |
| Granite | `ibm-granite/granite-4.1-8b` | 8B | Apache-2.0 | 2026-04 |
| gpt-oss | `openai/gpt-oss-20b` | 20B | Apache-2.0 | 2025-08 |
| OLMo 3 | `allenai/olmo-3-32b-think` | 32B | open | 2025-11 |

## Flags
- `qwen3-coder` denso pequeno **não existe** (só MoE a3B+). Para coder pequeno denso: `Seed-Coder-8B` ou `qwen2.5-coder-7b`.
- Gemma 3/4 são **gated** no HF (aceite de licença logado) — pesos abertos, mas exigem login.
