# linguistics-agent-eval

A tool-using linguistics research agent evaluated across multiple dimensions using a custom eval framework. The agent answers natural language questions about the world's languages by querying two real linguistic databases — WALS and Glottolog — and synthesizing across them.

Built to demonstrate agentic AI evaluation: tool-use accuracy, factual correctness against ground truth, and faithfulness to retrieved data.

---

## What This Is

The agent has access to seven tools backed by two real open-access linguistic databases:

- **WALS** (World Atlas of Language Structures) — typological features of 2,679 languages: word order, phonology, morphology, and more across 192 features
- **Glottolog 5.1** — language classification, family trees, and endangerment status for the world's languages

The most interesting capability is the **cross-source query**: finding languages that match a WALS typological feature *and* meet a Glottolog endangerment criterion — joining two heterogeneous databases via shared Glottocode identifiers. For example: *"which endangered SOV languages are there?"* requires WALS for word order data and Glottolog for endangerment status, joined on a shared identifier.

The eval framework tests three distinct dimensions:
- **Tool-use accuracy** — did the agent call the right tool(s) with the right parameters?
- **Factual accuracy** — do the agent's stated facts match ground truth from the databases?
- **Faithfulness** — is the agent's answer grounded in retrieved data, or does it add unsupported claims from training knowledge?

---

## Architecture

```
src/
├── tools.py              # 5 WALS query tools
├── glottolog_tools.py    # 2 Glottolog tools including cross-source query
├── agent.py              # Claude-powered reasoning loop with tool use
└── eval.py               # Three-dimensional eval suite with golden dataset

tests/
└── test_tools.py         # 19 pytest unit tests covering all WALS tools

data/                     # Downloaded datasets (not committed — see Setup)
```

### Tools

| Tool | Source | Description |
|------|--------|-------------|
| `lookup_language` | WALS | Language metadata: family, genus, coordinates, glottocode |
| `get_feature_info` | WALS | Feature description and possible values (e.g. what does feature 81A measure?) |
| `get_language_feature` | WALS | Value for a specific language on a specific feature |
| `find_languages_by_feature` | WALS | All languages with a given feature value, optionally filtered |
| `compare_languages` | WALS | Compare multiple languages on a single feature |
| `get_endangerment_status` | Glottolog | AES endangerment classification for a language |
| `find_endangered_languages_by_feature` | WALS + Glottolog | **Cross-source**: languages matching a WALS feature value AND an endangerment level |

Tools are deliberately kept separate rather than consolidated into a single function — this makes the agent's tool-selection decisions observable and evaluable. A production system would likely consolidate some of these, but separate tools make the capability boundaries visible for evaluation purposes.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Vince33/linguistics-agent-eval.git
cd linguistics-agent-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### 3. Download the data

**WALS** is accessed remotely via pycldf — no download needed.

**Glottolog** requires a local download (~45MB):

```bash
mkdir data
cd data
curl -L "https://github.com/glottolog/glottolog-cldf/archive/refs/tags/v5.1.zip" -o glottolog.zip
unzip glottolog.zip
cd ..
```

---

## Running the Agent

```python
from src.agent import run_agent

result = run_agent(
    "Which endangered languages have SOV word order?",
    verbose=True
)
print(result["answer"])
```

The agent returns:
- `answer` — the final natural language response
- `tool_calls` — every tool call made, with parameters
- `tool_results` — raw data returned from each tool

This three-part output implements the provenance principle from Hamel Husain's *"It's Hard to Eval"*: the answer is verifiable because the tool calls and raw results are surfaced alongside it, not hidden inside the agent loop.

---

## Running the Eval Suite

```bash
python3 -m src.eval
```

The eval suite runs 8 questions from a golden dataset against the agent and scores across three dimensions:

```
============================================================
EVAL SCORECARD
============================================================
Tool selection accuracy:  8/8 (100%)
Factual accuracy:         8/8 (100%)
Faithfulness:             8/8 (100%)
============================================================
```

### Eval dimensions

**Tool-use accuracy (deterministic):** checks whether the agent called the expected tool(s) for each question. Multi-tool questions (like "Is Navajo endangered and what is its word order?") require the agent to correctly identify that two separate tools are needed.

**Factual accuracy (deterministic, ground truth):** checks the agent's stated facts against values retrieved directly from WALS and Glottolog. Because both databases have structured, verifiable data, this check is purely deterministic — no LLM judge required.

**Faithfulness (LLM-as-judge):** uses DeepEval's `FaithfulnessMetric` with Claude Haiku 4.5 as judge to assess whether the agent's answer is grounded in retrieved tool results. Threshold: 0.7. This catches cases where the agent adds claims from training knowledge rather than retrieved data.

### A note on faithfulness scores

Not all passing questions scored 1.0 on faithfulness. The Ainu question scored 0.75 and the Navajo question scored 0.8 — both above threshold but worth watching across future runs. Lower scores on these questions likely reflect the agent adding accurate but unverified background context beyond what the tools returned. This is a real, observable failure mode in production agentic systems: the agent is correct but not fully grounded.

---

## Running the Unit Tests

```bash
pytest tests/test_tools.py -v
```

19 tests covering all five WALS tool functions: language lookup, feature info, language-feature queries, cross-linguistic comparison, and find-by-feature.

---

## Data Sources

- **WALS:** Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013. *The World Atlas of Language Structures Online.* Leipzig: Max Planck Institute for Evolutionary Anthropology. CC-BY 4.0.
- **Glottolog:** Hammarström, Harald & Forkel, Robert & Haspelmath, Martin & Bank, Sebastian. 2024. *Glottolog 5.1.* Leipzig: Max Planck Institute for Evolutionary Anthropology. CC-BY 4.0.