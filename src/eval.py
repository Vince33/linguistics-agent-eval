from dotenv import load_dotenv
load_dotenv()

import json
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import anthropic

from src.agent import run_agent
from src.tools import get_language_feature
from src.glottolog_tools import get_endangerment_status


# ─────────────────────────────────────────────
# Custom Claude judge for DeepEval
# ─────────────────────────────────────────────

class ClaudeHaiku(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "claude-haiku-4-5"


# ─────────────────────────────────────────────
# Golden dataset — ground truth from WALS/Glottolog
# ─────────────────────────────────────────────

GOLDEN_DATASET = [
    {
        "question": "What is the word order of Japanese?",
        "expected_tools": ["get_language_feature"],
        "expected_facts": {"language": "Japanese", "value": "SOV"},
        "ground_truth": get_language_feature("Japanese", "81A"),
    },
    {
        "question": "What is the word order of Mandarin?",
        "expected_tools": ["get_language_feature"],
        "expected_facts": {"language": "Mandarin", "value": "SVO"},
        "ground_truth": get_language_feature("Mandarin", "81A"),
    },
    {
        "question": "Is Ainu endangered?",
        "expected_tools": ["get_endangerment_status"],
        "expected_facts": {"endangerment_status": "nearly extinct"},
        "ground_truth": get_endangerment_status("ainu1240"),
    },
    {
        "question": "Is Japanese endangered?",
        "expected_tools": ["get_endangerment_status"],
        "expected_facts": {"endangerment_status": "not endangered"},
        "ground_truth": get_endangerment_status("nucl1643"),
    },
    {
        "question": "Do Japanese and Korean have the same word order?",
        "expected_tools": ["compare_languages"],
        "expected_facts": {"Japanese": "SOV", "Korean": "SOV"},
        "ground_truth": {
            "Japanese": get_language_feature("Japanese", "81A"),
            "Korean": get_language_feature("Korean", "81A"),
        },
    },
    {
        "question": "Is Navajo endangered and what is its word order?",
        "expected_tools": ["get_language_feature", "get_endangerment_status"],
        "expected_facts": {"word_order": "SOV", "endangerment": "not endangered"},
        "ground_truth": {
            "word_order": get_language_feature("Navajo", "81A"),
            "endangerment": get_endangerment_status("nava1243"),
        },
    },
    {
        "question": "How many endangered SOV languages are there?",
        "expected_tools": ["find_endangered_languages_by_feature"],
        "expected_facts": {"total_endangered_matches": 328},
        "ground_truth": {"total_endangered_matches": 328},
    },
    {
        "question": "What family does Swahili belong to?",
        "expected_tools": ["lookup_language"],
        "expected_facts": {"family": "Niger-Congo"},
        "ground_truth": {"family": "Niger-Congo", "name": "Swahili"},
    },
]


# ─────────────────────────────────────────────
# Dimension 1: Tool selection accuracy
# ─────────────────────────────────────────────

def eval_tool_selection(result: dict, expected_tools: list) -> dict:
    """
    Check whether the agent called the expected tools.
    Deterministic — no LLM judge needed.
    """
    tools_called = [c["tool"] for c in result["tool_calls"]]

    if len(expected_tools) > 1:
        # Multi-tool question — check all expected tools were called
        missing = [t for t in expected_tools if t not in tools_called]
        passed = len(missing) == 0
        return {
            "passed": passed,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "missing_tools": missing,
        }
    else:
        expected = expected_tools[0]
        passed = expected in tools_called
        return {
            "passed": passed,
            "tools_called": tools_called,
            "expected_tool": expected,
        }


# ─────────────────────────────────────────────
# Dimension 2: Factual accuracy
# ─────────────────────────────────────────────

def eval_factual_accuracy(result: dict, expected_facts: dict, ground_truth: dict) -> dict:
    """Check the agent's answer against ground truth from WALS/Glottolog.
    Deterministic — checks whether key facts appear in the agent's final answer.

    Note: This uses substring matching, which is fast and deterministic but
    can produce false positives (e.g. "not SOV" would pass a check for "SOV").
    For stronger factual checking, consider an LLM judge comparing the answer
    against ground_truth directly.
    """
    answer = result["answer"].lower()
    failures = []

    for key, expected_value in expected_facts.items():
        if isinstance(expected_value, str):
            if expected_value.lower() not in answer:
                failures.append({
                    "fact": key,
                    "expected": expected_value,
                    "found_in_answer": False
                })
        elif isinstance(expected_value, int):
            if str(expected_value) not in answer:
                failures.append({
                    "fact": key,
                    "expected": expected_value,
                    "found_in_answer": False
                })
        elif isinstance(expected_value, bool):
            # For boolean facts like "same word order"
            pass  # Handled by string facts above

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "ground_truth": ground_truth,
    }


# ─────────────────────────────────────────────
# Dimension 3: Faithfulness (LLM-as-judge)
# ─────────────────────────────────────────────

def eval_faithfulness(result: dict) -> dict:
    """
    Check whether the agent's answer is grounded in tool results.
    Uses DeepEval's FaithfulnessMetric with Claude Haiku as judge.
    """
    # Build context from tool results
    retrieval_context = [
        json.dumps(tr["result"], default=str)
        for tr in result["tool_results"]
    ]

    if not retrieval_context:
        return {
            "passed": None,
            "score": None,
            "reason": "No tool results to evaluate faithfulness against"
        }

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        retrieval_context=retrieval_context
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=ClaudeHaiku(),
        include_reason=True
    )

    metric.measure(test_case)

    return {
        "passed": metric.success,
        "score": metric.score,
        "reason": metric.reason,
    }

# ─────────────────────────────────────────────
# Dimension 4: Data consistency
# ─────────────────────────────────────────────

def eval_data_consistency(tool_results: list) -> dict:
    """
    Check whether tool results contain internal contradictions
    that could reasonably cause faithfulness failures.

    This dimension separates 'agent hallucinated' from 'source data
    was ambiguous' — a meaningful distinction for production AI QA.

    Currently detects: Glottolog AES numerical classification
    contradicting its own qualitative comment field.
    """
    issues = []
    for tr in tool_results:
        result = tr["result"]
        if "endangerment_status" in result and "comment" in result:
            status = result["endangerment_status"]
            comment = result.get("comment", "") or ""
            endangered_keywords = [
                "at risk", "threatened", "endangered",
                "vulnerable", "critically"
            ]
            comment_suggests_endangered = any(
                kw in comment.lower() for kw in endangered_keywords
            )
            if status == "not endangered" and comment_suggests_endangered:
                issues.append({
                    "tool": tr["tool"],
                    "issue": "AES classification contradicts comment text",
                    "classification": status,
                    "comment_excerpt": comment[:120]
                })

    return {
        "consistent": len(issues) == 0,
        "issues": issues
    }


# ─────────────────────────────────────────────
# Full eval runner
# ─────────────────────────────────────────────

def run_eval(verbose: bool = False) -> list:
    """
    Run the full evaluation suite against the golden dataset.
    Returns a list of results with scores across all three dimensions.
    """
    results = []

    for i, case in enumerate(GOLDEN_DATASET):
        print(f"\n[{i+1}/{len(GOLDEN_DATASET)}] {case['question']}")

        # Run the agent
        agent_result = run_agent(case["question"], verbose=verbose)

        # Dimension 1: Tool selection
        tool_eval = eval_tool_selection(
            agent_result,
            case["expected_tools"]
        )

        # Dimension 2: Factual accuracy
        factual_eval = eval_factual_accuracy(
            agent_result,
            case["expected_facts"],
            case["ground_truth"]
        )

        # Dimension 3: Faithfulness
        print("  Evaluating faithfulness...")
        faithfulness_eval = eval_faithfulness(agent_result)

        # Dimension 4: Data consistency
        consistency_eval = eval_data_consistency(agent_result["tool_results"])

        result = {
            "question": case["question"],
            "answer": agent_result["answer"],
            "tool_calls": agent_result["tool_calls"],
            "tool_selection": tool_eval,
            "factual_accuracy": factual_eval,
            "faithfulness": faithfulness_eval,
            "data_consistency": consistency_eval,
        }

        results.append(result)

        # Print summary for this case
        ts = "✓" if tool_eval["passed"] else "✗"
        fa = "✓" if factual_eval["passed"] else "✗"
        fh = "✓" if faithfulness_eval.get("passed") else "✗"
        dc = "✓" if consistency_eval["consistent"] else "⚠"
        score = faithfulness_eval.get("score", "N/A")
        print(f"  Tool selection: {ts} | Factual accuracy: {fa} | Faithfulness: {fh} ({score}) | Data consistency: {dc}")

        if not faithfulness_eval.get("passed") and not consistency_eval["consistent"]:
            print(f"  ⚠ Faithfulness failure may reflect inconsistent source data:")
            for issue in consistency_eval["issues"]:
                print(f"    {issue['issue']}: {issue['comment_excerpt']}")

    return results


def print_scorecard(results: list):
    """Print a summary scorecard of eval results."""
    total = len(results)
    tool_passed = sum(1 for r in results if r["tool_selection"]["passed"])
    factual_passed = sum(1 for r in results if r["factual_accuracy"]["passed"])
    faithful_passed = sum(
        1 for r in results
        if r["faithfulness"].get("passed") is True
    )
    consistent_passed = sum(
        1 for r in results
        if r["data_consistency"]["consistent"]
    )

    print("\n" + "="*60)
    print("EVAL SCORECARD")
    print("="*60)
    print(f"Tool selection accuracy:  {tool_passed}/{total} ({round(100*tool_passed/total)}%)")
    print(f"Factual accuracy:         {factual_passed}/{total} ({round(100*factual_passed/total)}%)")
    print(f"Faithfulness:             {faithful_passed}/{total} ({round(100*faithful_passed/total)}%)")
    print(f"Data consistency:         {consistent_passed}/{total} ({round(100*consistent_passed/total)}%)")
    print("="*60)

    failures = [r for r in results if not all([
        r["tool_selection"]["passed"],
        r["factual_accuracy"]["passed"],
        r["faithfulness"].get("passed", False)
    ])]

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"\n  Q: {r['question']}")
            if not r["tool_selection"]["passed"]:
                print(f"  ✗ Tool selection: called {r['tool_selection']['tools_called']}")
            if not r["factual_accuracy"]["passed"]:
                print(f"  ✗ Factual accuracy: {r['factual_accuracy']['failures']}")
            if not r["faithfulness"].get("passed", False):
                score = r["faithfulness"].get("score", "N/A")
                reason = r["faithfulness"].get("reason", "")[:100]
                print(f"  ✗ Faithfulness ({score}): {reason}")
                if not r["data_consistency"]["consistent"]:
                    print(f"  ⚠ Source data inconsistency detected:")
                    for issue in r["data_consistency"]["issues"]:
                        print(f"    → {issue['issue']}")
                        print(f"      AES classification: {issue['classification']}")
                        print(f"      Comment: {issue['comment_excerpt']}")
                    print(f"  → Faithfulness failure likely caused by inconsistent source data, not agent error")

if __name__ == "__main__":
    results = run_eval(verbose=False)
    print_scorecard(results)