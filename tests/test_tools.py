import pytest
from src.tools import (
    lookup_language,
    get_feature_info,
    get_language_feature,
    find_languages_by_feature,
    compare_languages,
)


class TestLookupLanguage:
    def test_exact_match_returns_correct_metadata(self):
        result = lookup_language("Japanese")
        assert result["id"] == "jpn"
        assert result["name"] == "Japanese"
        assert result["glottocode"] == "nucl1643"
        assert result["source"] == "WALS"

    def test_case_insensitive_match(self):
        result = lookup_language("japanese")
        assert result["id"] == "jpn"

    def test_unknown_language_returns_error(self):
        result = lookup_language("Klingon")
        assert "error" in result

    def test_ambiguous_name_returns_ambiguous(self):
        result = lookup_language("Arabic")
        assert "ambiguous" in result
        assert len(result["matches"]) > 1


class TestGetFeatureInfo:
    def test_exact_id_lookup(self):
        result = get_feature_info("81A")
        assert result["id"] == "81A"
        assert result["name"] == "Order of Subject, Object and Verb"
        assert result["area"] == "Word Order"

    def test_area_name_returns_primary_feature(self):
        result = get_feature_info("word order")
        assert result["id"] == "81A"

    def test_possible_values_included(self):
        result = get_feature_info("81A")
        codes = [v["code"] for v in result["possible_values"]]
        assert "SOV" in codes
        assert "SVO" in codes
        assert "VSO" in codes

    def test_unknown_feature_returns_error(self):
        result = get_feature_info("nonexistent feature xyz")
        assert "error" in result

    def test_partial_name_match_returns_result(self):
        # Tests the partial name/area match fallback path (lines 122-133)
        # "consonant" partially matches several feature names
        result = get_feature_info("consonant inventories")
        assert result["id"] == "1A"
        assert "error" not in result

    def test_ambiguous_partial_match_returns_ambiguous(self):
        # Tests the > 3 ambiguous return path (lines 127-132)
        # "order" matches many features by name
        result = get_feature_info("order of")
        assert "ambiguous" in result
        assert len(result["matches"]) > 3
    


class TestGetLanguageFeature:
    def test_japanese_word_order(self):
        result = get_language_feature("Japanese", "81A")
        assert result["language"] == "Japanese"
        assert result["value"] == "SOV"
        assert result["provenance"] == "WALS"

    def test_feature_by_area_name(self):
        result = get_language_feature("Japanese", "word order")
        assert result["value"] == "SOV"

    def test_source_citation_present(self):
        result = get_language_feature("Japanese", "81A")
        assert len(result["source_citation"]) > 0

    def test_missing_data_returns_error(self):
        # Not every language has data for every feature
        result = get_language_feature("Japanese", "1A")
        # Either returns a value or a clean error — never crashes
        assert "value" in result or "error" in result

    def test_invalid_feature_propagates_error(self):
        # Tests error propagation from get_feature_info (line 174)
        result = get_language_feature("Japanese", "nonexistent_xyz_feature")
        assert "error" in result


class TestFindLanguagesByFeature:
    def test_finds_sov_languages(self):
        result = find_languages_by_feature("81A", "SOV")
        assert result["value"] == "SOV"
        assert result["total_matches"] > 0
        assert len(result["languages"]) > 0

    def test_limit_respected(self):
        result = find_languages_by_feature("81A", "SOV", limit=5)
        assert len(result["languages"]) <= 5

    def test_invalid_value_returns_error(self):
        result = find_languages_by_feature("81A", "XYZ")
        assert "error" in result

    def test_provenance_present(self):
        result = find_languages_by_feature("81A", "SOV", limit=3)
        assert result["provenance"] == "WALS"

    def test_invalid_feature_propagates_error(self):
        # Tests error propagation from get_feature_info (line 228)
        result = find_languages_by_feature("nonexistent_xyz_feature", "SOV")
        assert "error" in result


class TestCompareLanguages:
    def test_japanese_korean_both_sov(self):
        result = compare_languages(["Japanese", "Korean"], "word order")
        values = {r["language"]: r["value"] for r in result["comparison"]}
        assert values["Japanese"] == "SOV"
        assert values["Korean"] == "SOV"

    def test_mandarin_is_svo(self):
        result = compare_languages(["Mandarin"], "word order")
        assert result["comparison"][0]["value"] == "SVO"

    def test_unknown_language_in_list_has_error(self):
        result = compare_languages(["Japanese", "Klingon"], "word order")
        errors = [r for r in result["comparison"] if r.get("error")]
        assert len(errors) == 1

    def test_invalid_feature_propagates_error(self):
        # Tests error propagation from get_feature_info (line 305)
        result = compare_languages(["Japanese", "Korean"], "nonexistent_xyz_feature")
        assert "error" in result