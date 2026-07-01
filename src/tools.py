from dotenv import load_dotenv
load_dotenv()

from pycldf import Dataset
from functools import lru_cache

WALS_URL = (
    "https://raw.githubusercontent.com/cldf-datasets/wals/v2020/cldf/"
    "StructureDataset-metadata.json"
)

@lru_cache(maxsize=1)
def _load_wals():
    """Load and cache the WALS dataset. Only fetches once per session."""
    ds = Dataset.from_metadata(WALS_URL)

    languages = {r["ID"]: dict(r) for r in ds["LanguageTable"]}
    parameters = {r["ID"]: dict(r) for r in ds["ParameterTable"]}
    codes = {}
    for r in ds["CodeTable"]:
        codes[r["ID"]] = dict(r)
    values = {}
    for r in ds["ValueTable"]:
        lang_id = r["Language_ID"]
        if lang_id not in values:
            values[lang_id] = {}
        values[lang_id][r["Parameter_ID"]] = dict(r)

    return languages, parameters, codes, values


def lookup_language(name: str) -> dict:
    """
    Look up a language by name. Returns metadata including family,
    genus, macroarea, glottocode, and coordinates.

    Args:
        name: The name of the language (e.g. 'Japanese', 'Swahili')

    Returns:
        dict with language metadata, or error message if not found
    """
    languages, _, _, _ = _load_wals()

    name_lower = name.lower()
    matches = [
        lang for lang in languages.values()
        if lang["Name"].lower() == name_lower
    ]

    if not matches:
        # Try partial match
        matches = [
            lang for lang in languages.values()
            if name_lower in lang["Name"].lower()
        ]

    if not matches:
        return {"error": f"No language found matching '{name}'"}

    if len(matches) == 1:
        lang = matches[0]
    else:
        return {
            "ambiguous": True,
            "matches": [{"id": l["ID"], "name": l["Name"]} for l in matches],
            "message": f"Multiple languages match '{name}'. Please be more specific."
        }

    return {
        "id": lang["ID"],
        "name": lang["Name"],
        "family": lang["Family"],
        "genus": lang["Genus"],
        "macroarea": lang["Macroarea"],
        "glottocode": lang["Glottocode"],
        "latitude": float(lang["Latitude"]) if lang["Latitude"] else None,
        "longitude": float(lang["Longitude"]) if lang["Longitude"] else None,
        "source": "WALS"
    }


def get_feature_info(feature_name: str) -> dict:
    """
    Look up a linguistic feature by name or ID.
    Returns the feature's description and possible values.

    Args:
        feature_name: Feature name (e.g. 'word order') or ID (e.g. '81A')

    Returns:
        dict with feature metadata and possible values
    """
    _, parameters, codes, _ = _load_wals()

    name_lower = feature_name.lower()

    # Try exact ID match first
    if feature_name.upper() in parameters:
        param = parameters[feature_name.upper()]
    else:
        # If query matches an area name exactly, default to shortest ID
        # in that area — this handles queries like "word order" or "phonology"
        exact_area_matches = [
            p for p in parameters.values()
            if name_lower == (p["Area"] or "").lower()
        ]
        if exact_area_matches:
            exact_area_matches.sort(key=lambda p: len(p["ID"]))
            param = exact_area_matches[0]
        else:
            # Try name match, then partial area match
            matches = [
                p for p in parameters.values()
                if name_lower in p["Name"].lower()
                or name_lower in (p["Area"] or "").lower()
            ]
            if not matches:
                return {"error": f"No feature found matching '{feature_name}'"}

            # Sort: name matches first, then by ID length
            matches.sort(key=lambda p: (
                0 if name_lower in p["Name"].lower() else 1,
                len(p["ID"])
            ))

            if len(matches) > 3:
                return {
                    "ambiguous": True,
                    "matches": [{"id": p["ID"], "name": p["Name"]} for p in matches[:10]],
                    "message": f"Multiple features match '{feature_name}'. Please be more specific."
                }
            param = matches[0]

    # Get possible values for this feature
    feature_codes = [
        {"code": c["Name"], "description": c["Description"]}
        for c in codes.values()
        if c["Parameter_ID"] == param["ID"]
    ]

    return {
        "id": param["ID"],
        "name": param["Name"],
        "area": param["Area"],
        "possible_values": feature_codes,
        "source": "WALS"
    }


def get_language_feature(language_name: str, feature_name: str) -> dict:
    """
    Get the value of a specific linguistic feature for a specific language.

    Args:
        language_name: Name of the language (e.g. 'Japanese')
        feature_name: Feature name or ID (e.g. 'word order' or '81A')

    Returns:
        dict with the feature value and human-readable label, with provenance
    """
    languages, parameters, codes, values = _load_wals()

    # Resolve language
    lang_result = lookup_language(language_name)
    if "error" in lang_result or "ambiguous" in lang_result:
        return lang_result

    lang_id = lang_result["id"]

    # Resolve feature
    feature_result = get_feature_info(feature_name)
    if "error" in feature_result or "ambiguous" in feature_result:
        return feature_result

    feature_id = feature_result["id"]

    # Look up the value
    lang_values = values.get(lang_id, {})
    if feature_id not in lang_values:
        return {
            "error": f"No data for feature '{feature_result['name']}' "
                     f"in language '{lang_result['name']}'"
        }

    value_record = lang_values[feature_id]
    code = codes.get(value_record["Code_ID"], {})

    return {
        "language": lang_result["name"],
        "language_id": lang_id,
        "feature": feature_result["name"],
        "feature_id": feature_id,
        "value": code.get("Name", value_record["Value"]),
        "description": code.get("Description", ""),
        "source_citation": value_record.get("Source", []),
        "provenance": "WALS",
        "raw_value": value_record["Value"]
    }


def find_languages_by_feature(
    feature_name: str,
    value: str,
    family: str = None,
    macroarea: str = None,
    limit: int = 20
) -> dict:
    """
    Find all languages with a specific value for a given feature.
    Optionally filter by language family or macroarea.

    Args:
        feature_name: Feature name or ID (e.g. 'word order' or '81A')
        value: The value to filter by (e.g. 'SOV', 'SVO')
        family: Optional language family filter (e.g. 'Austronesian')
        macroarea: Optional macroarea filter (e.g. 'Africa', 'Eurasia')
        limit: Maximum number of results to return (default 20)

    Returns:
        dict with matching languages and provenance
    """
    languages, _, codes, values = _load_wals()

    # Resolve feature
    feature_result = get_feature_info(feature_name)
    if "error" in feature_result or "ambiguous" in feature_result:
        return feature_result

    feature_id = feature_result["id"]

    # Find matching code ID for the requested value
    target_code = None
    for code_id, code in codes.items():
        if (code["Parameter_ID"] == feature_id and
                code["Name"].lower() == value.lower()):
            target_code = code
            break

    if not target_code:
        possible = [
            c["Name"] for c in codes.values()
            if c["Parameter_ID"] == feature_id
        ]
        return {
            "error": f"Value '{value}' not found for feature "
                     f"'{feature_result['name']}'. "
                     f"Possible values: {', '.join(possible)}"
        }

    # Find all languages with this value
    lang_matches = []
    for lang_id, lang_values in values.items():
        if feature_id not in lang_values:
            continue
        if lang_values[feature_id]["Code_ID"] != target_code["ID"]:
            continue

        lang = languages.get(lang_id)
        if not lang:
            continue

        # Apply optional filters
        if family and lang["Family"] and family.lower() not in lang["Family"].lower():
            continue
        if macroarea and lang["Macroarea"] and macroarea.lower() not in lang["Macroarea"].lower():
            continue

        lang_matches.append({
            "name": lang["Name"],
            "id": lang_id,
            "family": lang["Family"],
            "macroarea": lang["Macroarea"],
            "glottocode": lang["Glottocode"]
        })

    return {
        "feature": feature_result["name"],
        "feature_id": feature_id,
        "value": target_code["Name"],
        "total_matches": len(lang_matches),
        "languages": lang_matches[:limit],
        "truncated": len(lang_matches) > limit,
        "filters_applied": {
            "family": family,
            "macroarea": macroarea
        },
        "provenance": "WALS"
    }


def compare_languages(language_names: list, feature_name: str) -> dict:
    """
    Compare multiple languages on a single linguistic feature.

    Args:
        language_names: List of language names (e.g. ['Japanese', 'Korean', 'Mandarin'])
        feature_name: Feature name or ID to compare on

    Returns:
        dict with each language's value for the feature, with provenance
    """
    feature_result = get_feature_info(feature_name)
    if "error" in feature_result or "ambiguous" in feature_result:
        return feature_result

    results = []
    for name in language_names:
        result = get_language_feature(name, feature_name)
        results.append({
            "language": name,
            "value": result.get("value"),
            "description": result.get("description"),
            "error": result.get("error")
        })

    return {
        "feature": feature_result["name"],
        "feature_id": feature_result["id"],
        "comparison": results,
        "provenance": "WALS"
    }