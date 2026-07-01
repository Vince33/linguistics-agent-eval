from dotenv import load_dotenv
load_dotenv()

import anthropic
import json
from src.tools import (
    lookup_language,
    get_feature_info,
    get_language_feature,
    find_languages_by_feature,
    compare_languages,
)
from src.glottolog_tools import (
    get_endangerment_status,
    find_endangered_languages_by_feature,
)

client = anthropic.Anthropic()

# Tool definitions — these tell Claude what tools exist and how to call them
TOOLS = [
    {
        "name": "lookup_language",
        "description": (
            "Look up a language by name. Returns metadata including family, "
            "genus, macroarea, glottocode, and coordinates. Use this first "
            "to confirm a language exists in WALS before querying its features."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the language (e.g. 'Japanese', 'Swahili')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_feature_info",
        "description": (
            "Look up a linguistic feature by name or ID. Returns the feature's "
            "description, area (e.g. Phonology, Word Order), and all possible "
            "values. Use this to understand what a feature measures before "
            "querying it for a specific language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "Feature name (e.g. 'consonant inventories') or WALS ID (e.g. '1A')"
                }
            },
            "required": ["feature_name"]
        }
    },
    {
        "name": "get_language_feature",
        "description": (
            "Get the value of a specific linguistic feature for a specific language. "
            "Returns the human-readable value with source citation. "
            "This is the primary tool for answering questions about a single "
            "language's typological properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language_name": {
                    "type": "string",
                    "description": "Name of the language (e.g. 'Japanese')"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Feature name or WALS ID (e.g. 'word order' or '81A')"
                }
            },
            "required": ["language_name", "feature_name"]
        }
    },
    {
        "name": "find_languages_by_feature",
        "description": (
            "Find all languages in WALS with a specific value for a given feature. "
            "Optionally filter by language family or macroarea. Use this for "
            "questions like 'which languages have SOV word order' or "
            "'which Austronesian languages lack tone'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "Feature name or WALS ID"
                },
                "value": {
                    "type": "string",
                    "description": "The value to filter by (e.g. 'SOV', 'SVO')"
                },
                "family": {
                    "type": "string",
                    "description": "Optional language family filter (e.g. 'Austronesian')"
                },
                "macroarea": {
                    "type": "string",
                    "description": "Optional macroarea filter (e.g. 'Africa', 'Eurasia')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 20)"
                }
            },
            "required": ["feature_name", "value"]
        }
    },
    {
        "name": "compare_languages",
        "description": (
            "Compare multiple languages on a single linguistic feature. "
            "Use this for questions like 'do Japanese and Korean have the "
            "same word order' or 'how do these three languages differ in "
            "their consonant inventories'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of language names to compare"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Feature name or WALS ID to compare on"
                }
            },
            "required": ["language_names", "feature_name"]
        }
    },
    {
        "name": "get_endangerment_status",
        "description": (
            "Get the endangerment status of a language from Glottolog. "
            "Requires a Glottocode identifier, which can be obtained from "
            "lookup_language. Returns the AES (Agglomerated Endangerment Status) "
            "which ranges from 'not endangered' to 'extinct', with source citations. "
            "Use this when asked about whether a language is endangered, threatened, "
            "or extinct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "glottocode": {
                    "type": "string",
                    "description": "The Glottocode identifier (e.g. 'nucl1643' for Japanese)"
                }
            },
            "required": ["glottocode"]
        }
    },
    {
        "name": "find_endangered_languages_by_feature",
        "description": (
            "Cross-source query: find languages that have a specific WALS "
            "typological feature value AND are endangered according to Glottolog. "
            "This joins WALS and Glottolog data via the shared glottocode identifier. "
            "Use this for questions like 'which endangered languages have SOV word order' "
            "or 'which extinct languages lacked tone'. "
            "This is the most powerful tool for questions combining typology and endangerment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {
                    "type": "string",
                    "description": "WALS feature name or ID (e.g. 'word order' or '81A')"
                },
                "value": {
                    "type": "string",
                    "description": "The WALS feature value (e.g. 'SOV', 'SVO')"
                },
                "endangerment_levels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of endangerment levels to include. Options: 'threatened', 'shifting', 'moribund', 'nearly extinct', 'extinct'. Defaults to all endangered levels."
                },
                "family": {
                    "type": "string",
                    "description": "Optional language family filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 20)"
                }
            },
            "required": ["feature_name", "value"]
        }
    },
]

TOOL_FUNCTIONS = {
    "lookup_language": lookup_language,
    "get_feature_info": get_feature_info,
    "get_language_feature": get_language_feature,
    "find_languages_by_feature": find_languages_by_feature,
    "compare_languages": compare_languages,
    "get_endangerment_status": get_endangerment_status,
    "find_endangered_languages_by_feature": find_endangered_languages_by_feature,
}

SYSTEM_PROMPT = """You are a linguistics research assistant with access to two
linguistic databases: the World Atlas of Language Structures (WALS) for typological
features, and Glottolog for language classification and endangerment status.

When answering questions:
- Always use your tools to retrieve data rather than relying on your own knowledge
- Cite the source of your data (WALS feature ID and Glottolog source citations when available)
- Be precise about what the data shows and what it does not show
- If data is missing for a language or feature, say so clearly
- Show your reasoning — explain what you looked up and why
- For questions combining typology and endangerment, use find_endangered_languages_by_feature

You have access to data on 2,679 languages across 192 typological features (WALS)
and endangerment status for thousands of languages (Glottolog 5.1)."""


def run_agent(question: str, verbose: bool = False) -> dict:
    """
    Run the linguistics research agent on a question.

    Args:
        question: A natural language question about language typology
        verbose: If True, print tool calls and results as they happen

    Returns:
        dict with the final answer, tool calls made, and raw tool results
    """
    messages = [{"role": "user", "content": question}]
    tool_calls_made = []
    tool_results = []

    MAX_TURNS = 10
    turn = 0

    while turn < MAX_TURNS:
        turn += 1
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"\n[Agent] Stop reason: {response.stop_reason}")

        # If Claude is done, extract the final text answer
        if response.stop_reason == "end_turn":
            final_answer = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return {
                "question": question,
                "answer": final_answer,
                "tool_calls": tool_calls_made,
                "tool_results": tool_results,
            }

        # Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add Claude's response to the message history
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call
            tool_results_for_turn = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(f"\n[Tool Call] {tool_name}({tool_input})")

                # Record the tool call
                tool_calls_made.append({
                    "tool": tool_name,
                    "input": tool_input
                })

                # Execute the tool
                tool_fn = TOOL_FUNCTIONS.get(tool_name)
                if tool_fn:
                    result = tool_fn(**tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                if verbose:
                    print(f"[Tool Result] {json.dumps(result, indent=2)}")

                # Record the result
                tool_results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": result
                })

                tool_results_for_turn.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str)
                })

            # Add tool results to message history
            messages.append({
                "role": "user",
                "content": tool_results_for_turn
            })

    # Safety fallback if MAX_TURNS exceeded
    return {
        "question": question,
        "answer": "Agent exceeded maximum turns without completing.",
        "tool_calls": tool_calls_made,
        "tool_results": tool_results,
    }