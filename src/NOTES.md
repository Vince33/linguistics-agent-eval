# linguistics-agent-eval — Notes

## Finding — Glottolog data inconsistency: AES classification contradicts comment field

### Background

The eval suite includes a question about Navajo's endangerment status. Glottolog's
AES (Agglomerated Endangerment Status) system assigns each language a numerical
classification from 1 (not endangered) to 6 (extinct), alongside a qualitative
comment field citing the source assessment.

For Navajo, the AES numerical value is `1` — classified as `not endangered` — with
the comment: "Diné Bizaad (Navajo) = At risk (20 percent certain, based on the
evidence available)."

### Finding

The structured `endangerment_status` field and the qualitative `comment` field point
in different directions. The AES classification says `not endangered`; the comment
says "At risk." When the agent answers a question about Navajo's endangerment, it
reads both fields and reports "at risk" — which contradicts the structured
classification returned by the tool, triggering a faithfulness failure at 0.67
(below the 0.7 threshold).

This is not an agent error. The agent is accurately reporting what Glottolog's own
comment says. The issue is that Glottolog's two representations of the same
information disagree.

### Root cause

Glottolog's AES system aggregates assessments from multiple sources (EGIDS, UNESCO,
ElCat) and assigns a single numerical classification. The comment field preserves
the original source language, which may reflect a different scale, different
certainty level, or different assessment date than the aggregated AES score. These
two fields are not guaranteed to agree.

### Design response

Rather than lowering the faithfulness threshold to accommodate this case, a fourth
evaluation dimension was added: **data consistency**. This dimension checks whether
tool results contain internal contradictions that could reasonably cause faithfulness
failures — specifically, whether Glottolog's AES classification contradicts its own
comment text. When a faithfulness failure co-occurs with a data consistency flag,
the scorecard notes it explicitly:

```
⚠ Source data inconsistency detected:
  → AES classification contradicts comment text
    AES classification: not endangered
    Comment: Diné Bizaad (Navajo) = At risk (20 percent certain...)
→ Faithfulness failure likely caused by inconsistent source data, not agent error
```

This preserves the 0.7 faithfulness threshold for genuine agent failures while
surfacing the underlying data quality issue as a named, separate finding.

### Practical implication

Any production system using Glottolog endangerment data should decide explicitly
which field to treat as authoritative — the structured AES numerical classification,
or the qualitative comment text — and surface that decision clearly to users. Passing
both to a language model without disambiguation creates ambiguity the model will
resolve in unpredictable ways.

---

## Finding — Faithfulness score variance across eval runs

### Observation

Across two successive full eval runs, faithfulness failures occurred on different
questions:

- Run 1: Navajo failed faithfulness (0.67), Ainu passed (0.75)
- Run 2: Ainu failed faithfulness (0.67), Navajo passed (0.80)

Both failures were on questions where faithfulness scores were close to the 0.7
threshold. The underlying agent answers were substantively correct in both cases —
the failures reflected the LLM judge applying its scoring rubric slightly differently
across runs, not genuine changes in agent behavior.

### What this demonstrates

The faithfulness judge itself introduces variance. A single eval run is not sufficient
to distinguish a genuine quality regression from normal judge variance. This is the
"separating model variance from quality regressions" problem applied to the eval
framework itself, not just the agent being evaluated.

### Implication for eval design

Faithfulness scores near the threshold (0.7-0.8) should be treated as uncertain
rather than definitive. A meaningful regression signal requires either:

- Multiple runs and a statistical drop in pass rate (e.g. from 90% to 60% across
  20 runs), rather than a single failing run
- A larger margin above threshold on the baseline (e.g. consistently scoring 0.9+)
  so that normal variance doesn't reach the failure boundary
- A deterministic check for the specific failure mode, rather than relying on
  the probabilistic LLM judge

The data consistency dimension added in response to the Navajo finding is an example
of the third approach: replacing an uncertain LLM judgment with a deterministic check
for a specific, known failure mode.

---

## Design decision — separate tools vs consolidated function

Seven tools are defined across two data sources (WALS and Glottolog). A production
system would likely consolidate several of these — for example, `lookup_language`
and `get_endangerment_status` could be combined into a single language metadata
function that returns both WALS and Glottolog data in one call.

The tools are kept separate here deliberately: granular tools make the agent's
tool-selection decisions visible and evaluable. A single consolidated tool would
make it impossible to evaluate whether the agent correctly identified that a question
required cross-source data — one of the three evaluation dimensions in this project.
A consolidated tool would also make it harder to trace which source contributed
which part of a given answer, reducing provenance clarity.

This tradeoff is acknowledged in the codebase. The separation serves evaluation
purposes; production engineering would consolidate.

---

## Design decision — tool descriptions as prompt engineering

The agent achieved 8/8 tool selection accuracy across all eval runs. This is
partly attributable to deliberate, specific language in the tool descriptions passed
to the model. Examples:

- `lookup_language`: "Use this first to confirm a language exists in WALS before
  querying its features" — explicitly guides sequencing
- `get_endangerment_status`: "Requires a Glottocode identifier, which can be obtained
  from lookup_language" — teaches the two-step pattern explicitly
- `find_endangered_languages_by_feature`: "This is the most powerful tool for
  questions combining typology and endangerment" — signals priority

Tool descriptions are not just documentation — they are the agent's primary
decision-making resource for tool selection. The quality of these descriptions
directly affects tool-use accuracy, and they should be treated with the same
attention given to system prompts.

---

## Design decision — provenance as a first-class output

The `run_agent` function returns three artifacts for every question: the final
answer, the list of tool calls made (with parameters), and the raw tool results.
This three-part output is deliberate, following the principle in Hamel Husain's
*"It's Hard to Eval"* (June 2026): artifacts that are hard to verify are often
hard for users to trust. Surfacing the tool calls and raw data alongside the
final answer makes the answer checkable — a reader can verify that the stated
facts appear in the retrieved data, and identify where the agent may have added
context beyond what the tools returned.

This structure also made the eval framework significantly easier to build: having
raw tool results available as a first-class artifact meant faithfulness evaluation
required no additional instrumentation of the agent loop.

---

## Eval baseline — final results

Eight questions from a golden dataset, scored across four dimensions:

```
Tool selection accuracy:  8/8 (100%)
Factual accuracy:         8/8 (100%)
Faithfulness:             7/8 (88%) — variance across runs; see finding above
Data consistency:         7/8 (88%) — Navajo AES contradiction; see finding above
```

The one consistent signal across both runs: Navajo's data consistency flag (`⚠`)
appears reliably because it is a deterministic check against a real, stable data
quality issue in Glottolog 5.1. The faithfulness failures vary because they depend
on a probabilistic LLM judge operating near the threshold boundary.