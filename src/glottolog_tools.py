from dotenv import load_dotenv
load_dotenv()

from pycldf import Dataset
from functools import lru_cache
import os

GLOTTOLOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "glottolog-cldf-5.1", "cldf", "cldf-metadata.json"
)

AES_LABELS = {
    "aes-not_endangered": "not endangered",
    "aes-threatened": "threatened",
    "aes-shifting": "shifting",
    "aes-moribund": "moribund",
    "aes-nearly_extinct": "nearly extinct",
    "aes-extinct": "extinct",
}

AES_NUMERICAL = {
    "1": "not endangered",
    "2": "threatened",
    "3": "shifting",
    "4": "moribund",
    "5": "nearly extinct",
    "6": "extinct",
}


@lru_cache(maxsize=1)
def _load_glottolog():
    """Load and cache the Glottolog dataset. Only loads once per session."""
    ds = Dataset.from_metadata(GLOTTOLOG_PATH)

    languages = {r["ID"]: dict(r) for r in ds["LanguageTable"]}

    # Build values index keyed by language_id -> parameter_id -> value record
    values = {}
    for r in ds["ValueTable"]:
        lang_id = r["Language_ID"]
        if lang_id not in values:
            values[lang_id] = {}
        values[lang_id][r["Parameter_ID"]] = dict(r)

    return languages, values


def get_endangerment_status(glottocode: str) -> dict:
    """
    Get the endangerment status of a language from Glottolog.

    Args:
        glottocode: The Glottocode identifier (e.g. 'nucl1643' for Japanese)

    Returns:
        dict with endangerment status, description, and source
    """
    languages, values = _load_glottolog()

    if glottocode not in values:
        return {"error": f"No Glottolog data found for glottocode '{glottocode}'"}

    lang_values = values[glottocode]
    lang = languages.get(glottocode, {})

    # Get AES value
    aes_record = lang_values.get("aes")
    if not aes_record:
        return {
            "error": f"No endangerment data for glottocode '{glottocode}'"
        }

    status = AES_NUMERICAL.get(aes_record["Value"], "unknown")
    code_id = aes_record.get("Code_ID", "")
    description = ""

    # Get classification
    classification_record = lang_values.get("classification")
    classification = classification_record["Value"] if classification_record else None

    # Get category
    category_record = lang_values.get("category")
    category = category_record["Value"] if category_record else None

    return {
        "glottocode": glottocode,
        "name": lang.get("Name"),
        "endangerment_status": status,
        "numerical_value": aes_record["Value"],
        "comment": aes_record.get("Comment"),
        "classification": classification,
        "category": category,
        "source": "Glottolog 5.1"
    }


def find_endangered_languages_by_feature(
    feature_name: str,
    value: str,
    endangerment_levels: list = None,
    family: str = None,
    limit: int = 20
) -> dict:
    """
    Cross-source query: find languages with a specific WALS feature value
    that are also endangered according to Glottolog.

    This is the primary cross-source tool — it joins WALS typological data
    with Glottolog endangerment data via the shared glottocode identifier.

    Args:
        feature_name: WALS feature name or ID (e.g. 'word order' or '81A')
        value: The WALS feature value (e.g. 'SOV')
        endangerment_levels: List of endangerment levels to include
            Options: 'threatened', 'shifting', 'moribund', 'nearly extinct', 'extinct'
            Defaults to all endangered (excludes 'not endangered')
        family: Optional language family filter
        limit: Maximum results to return

    Returns:
        dict with matching languages, their WALS feature value,
        and their Glottolog endangerment status
    """
    from src.tools import find_languages_by_feature

    # Default to all endangered levels
    if endangerment_levels is None:
        endangerment_levels = [
            "threatened", "shifting", "moribund", "nearly extinct", "extinct"
        ]

    # Get WALS languages with this feature value
    wals_result = find_languages_by_feature(
        feature_name=feature_name,
        value=value,
        family=family,
        limit=500  # Get a large set to filter from
    )

    if "error" in wals_result:
        return wals_result

    _, glottolog_values = _load_glottolog()

    # Filter to endangered languages using Glottolog data
    matches = []
    no_glottolog_data = 0

    for lang in wals_result["languages"]:
        glottocode = lang.get("glottocode")
        if not glottocode:
            no_glottolog_data += 1
            continue

        lang_values = glottolog_values.get(glottocode)
        if not lang_values:
            no_glottolog_data += 1
            continue

        aes_record = lang_values.get("aes")
        if not aes_record:
            no_glottolog_data += 1
            continue

        status = AES_NUMERICAL.get(aes_record["Value"], "unknown")
        if status not in endangerment_levels:
            continue

        matches.append({
            "name": lang["name"],
            "wals_id": lang["id"],
            "glottocode": glottocode,
            "family": lang["family"],
            "wals_feature": wals_result["feature"],
            "wals_value": wals_result["value"],
            "endangerment_status": status,
            "endangerment_comment": aes_record.get("Comment", ""),
            "sources": ["WALS", "Glottolog 5.1"]
        })

    return {
        "feature": wals_result["feature"],
        "feature_id": wals_result["feature_id"],
        "wals_value": wals_result["value"],
        "endangerment_levels_queried": endangerment_levels,
        "total_wals_matches": wals_result["total_matches"],
        "total_endangered_matches": len(matches),
        "languages": matches[:limit],
        "truncated": len(matches) > limit,
        "languages_without_glottolog_data": no_glottolog_data,
        "filters_applied": {"family": family},
        "sources": ["WALS", "Glottolog 5.1"],
        "provenance": "Cross-source query: WALS joined with Glottolog via glottocode"
    }