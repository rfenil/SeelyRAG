"""Stage 5 query understanding.

build-plan section 7.1. The deterministic pass is tested as the contract,
because it is what actually runs: it needs no key, and the LLM pass may only
*add* to it.

The tests below name real installer phrasings rather than synthetic ones. Each
came from working through the corpus's own vocabulary.
"""

from __future__ import annotations

from typing import Any

import pytest

from seeley_rag.retrieve.query import (
    LLM_SYSTEM_PROMPT,
    Understanding,
    understand,
    understand_deterministic,
)


class TestProductFamily:
    """Resolved from the same lexicon Stage 2 used to label 13,156 pages."""

    @pytest.mark.parametrize(
        ("query", "family"),
        [
            ("the ducted heater is throwing E:04", "DGH"),
            ("Braemar evaporative cooler water pump not priming", "EVAP"),
            ("VRF outdoor unit high discharge temperature", "VRF"),
            ("reverse cycle split system not cooling", "RC"),
            # "controller" is a CONTROLS alias and MagIQtouch is listed under
            # both families; the controls line is the more specific answer.
            ("MagIQtouch controller setup", "CONTROLS"),
        ],
    )
    def test_family_is_inferred_from_vocabulary(self, query: str, family: str) -> None:
        """Installer phrasing, not portal category names."""
        assert understand_deterministic(query).product_family == family

    def test_longest_pattern_wins(self) -> None:
        """ "VRF reverse cycle" contains RC's "reverse cycle".

        First-match ordering mislabelled 1,622 pages in Stage 2 before this rule
        was applied there; the same trap exists here.
        """
        assert understand_deterministic("VRF reverse cycle fault").product_family == "VRF"

    def test_unknown_when_nothing_matches(self) -> None:
        """A wrong family is worse than an absent one -- it is soft-boosted on."""
        assert understand_deterministic("how do I clean the filter").product_family == "UNKNOWN"


class TestModelSeries:
    """Model codes, including the suffixed forms installers actually type."""

    def test_bare_code_is_found(self) -> None:
        """The lexicon lists TQ."""
        assert "TQ" in understand_deterministic("TQ service guide").model_series

    def test_suffixed_code_resolves_to_its_prefix(self) -> None:
        """The lexicon lists TQ; an installer writes the size on the end.

        Without this, "manifold pressure for a TQ5" got no family at all.
        """
        parsed = understand_deterministic("what is the manifold gas pressure for a TQ5")
        assert parsed.model_series == ["TQ"]
        assert parsed.product_family == "DGH"

    def test_a_suffixed_vrf_code_resolves(self) -> None:
        """Same rule across families."""
        assert understand_deterministic("MCMX3 outdoor unit").product_family == "VRF"

    def test_ordinary_words_are_not_model_codes(self) -> None:
        """The prefix rule must require a known code, not any letters+digits."""
        assert understand_deterministic("check within 24 hours").model_series == []

    def test_explicit_model_enables_hard_filtering(self) -> None:
        """The one case section 7.1 allows a hard filter."""
        assert understand_deterministic("TQ5 wiring").model_explicit is True

    def test_a_vague_reference_does_not(self) -> None:
        """ "the ducted heater" names no model, so nothing may be filtered."""
        assert understand_deterministic("the ducted heater is faulty").model_explicit is False


class TestFaultCodes:
    """Codes are the field where a hallucination would be worst."""

    @pytest.mark.parametrize(
        ("query", "code"),
        [
            ("the heater is throwing E:04", "E04"),
            ("fault code 30 on the TQ", "FC30"),
            ("TQ heater FC7 what do I check", "FC07"),
            ("getting FC 53 on the display", "FC53"),
        ],
    )
    def test_codes_are_extracted_and_normalised(self, query: str, code: str) -> None:
        """Every printed spelling has to reach the same lookup key."""
        assert code in understand_deterministic(query).fault_codes

    def test_fc7_and_fault_code_7_agree(self) -> None:
        """The manuals print "Fault Code 07"; installers say "FC7"."""
        assert (
            understand_deterministic("FC7 fault").fault_codes
            == understand_deterministic("fault code 7").fault_codes
        )

    def test_no_codes_in_an_ordinary_question(self) -> None:
        """A false code is pinned ahead of retrieval, so precision matters."""
        assert understand_deterministic("how do I clean the filter").fault_codes == []


class TestIntent:
    """Recorded so the eval can slice accuracy by question type."""

    @pytest.mark.parametrize(
        ("query", "intent"),
        [
            ("the heater won't ignite", "fault_diagnosis"),
            ("what part number is the pump", "parts"),
            ("clearance required to install the unit", "installation"),
            ("what is the manifold gas pressure", "specification"),
            ("how often should I clean the filter", "maintenance"),
            ("tell me about Braemar", "general"),
        ],
    )
    def test_intent_is_classified(self, query: str, intent: str) -> None:
        """Marker vocabulary, most specific first."""
        assert understand_deterministic(query).intent == intent

    def test_a_named_code_makes_it_a_fault_question(self) -> None:
        """ "the ducted heater is throwing E:04" has no marker vocabulary at all.

        A named code IS a fault question whatever words surround it.
        """
        assert understand_deterministic("the ducted heater is throwing E:04").intent == (
            "fault_diagnosis"
        )


class TestDiagramIntent:
    """Drives whether the generator surfaces a page image."""

    @pytest.mark.parametrize(
        "query",
        [
            "show me the wiring diagram",
            "what does the exploded view look like",
            "where is the flame sensor",
            "which terminal does the blue wire go to",
        ],
    )
    def test_diagram_requests_are_detected(self, query: str) -> None:
        """The corpus is 14% diagram-heavy; these are the queries it serves."""
        assert understand_deterministic(query).wants_diagram is True

    def test_ordinary_questions_do_not_want_a_diagram(self) -> None:
        """Surfacing an image for every answer would be noise."""
        assert understand_deterministic("what is the gas pressure").wants_diagram is False


class FakeClient:
    """An OpenAI-shaped client, matching the configured default provider.

    The shape has to follow ``generate.provider``: query understanding now goes
    through :mod:`seeley_rag.llm`, which dispatches on configuration rather than
    on the client it is handed.
    """

    def __init__(self, payload: str) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Completions:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                message = type("Message", (), {"content": payload})()
                return type(
                    "Response", (), {"choices": [type("Choice", (), {"message": message})()]}
                )()

        self.chat = type("Chat", (), {"completions": Completions()})()


class TestLlmEnrichment:
    """The LLM may add, never overwrite what regexes established."""

    def test_the_rewrite_may_not_name_a_product_the_query_did_not(self) -> None:
        """Guessing a family turns an ambiguous question into a confident wrong one.

        Measured: on a bare "fc7" the rewrite invented "gas ducted heating" and on
        "fc 7" it invented "evaporative cooler" -- the same question, opposite
        guesses, each then retrieved for confidently. Ambiguity is the code
        lookup's job; the rewrite expands symptom vocabulary only.
        """
        assert (
            "NEVER name a product family, product type or model the query did not name"
            in LLM_SYSTEM_PROMPT
        )

    def test_llm_supplies_a_rewritten_query(self) -> None:
        """The one field the deterministic pass cannot produce."""
        client = FakeClient(
            '{"intent": "fault_diagnosis", "wants_diagram": false, '
            '"rewritten_query": "TQ gas ducted heater FC7 ignition failure"}'
        )
        parsed = understand("TQ FC7", client=client)
        assert parsed.rewritten_query == "TQ gas ducted heater FC7 ignition failure"
        assert parsed.source == "deterministic+openai"
        assert parsed.search_text == parsed.rewritten_query

    def test_llm_cannot_overwrite_extracted_codes(self) -> None:
        """A hallucinated E:05 in place of E:04 would be pinned into context.

        The model is never asked for codes and its output cannot reach them.
        """
        client = FakeClient(
            '{"intent": "general", "wants_diagram": false, '
            '"rewritten_query": "something else", "fault_codes": ["E05"]}'
        )
        assert understand("TQ FC7", client=client).fault_codes == ["FC07"]

    def test_llm_cannot_overwrite_the_product_family(self) -> None:
        """The lexicon is more reliable than a model here, and cheaper."""
        client = FakeClient(
            '{"intent": "general", "wants_diagram": false, '
            '"rewritten_query": "x", "product_family": "EVAP"}'
        )
        assert understand("the ducted heater is faulty", client=client).product_family == "DGH"

    def test_diagram_intent_is_a_union_not_a_replacement(self) -> None:
        """Either signal is enough; the regex must not be overridden to False."""
        client = FakeClient('{"intent": "general", "wants_diagram": false, "rewritten_query": "x"}')
        assert understand("show me the wiring diagram", client=client).wants_diagram is True

    def test_a_malformed_response_is_not_fatal(self) -> None:
        """The deterministic understanding is already usable."""
        parsed = understand("TQ FC7", client=FakeClient("not json at all"))
        assert parsed.fault_codes == ["FC07"]
        assert parsed.source == "deterministic"

    def test_a_raising_client_is_not_fatal(self) -> None:
        """A router outage must not take retrieval down with it."""

        class Broken:
            class chat:  # noqa: N801 - mirrors the SDK's shape
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        raise RuntimeError("router down")

        parsed = understand("TQ FC7", client=Broken())
        assert parsed.fault_codes == ["FC07"]
        assert parsed.source == "deterministic"

    def test_the_router_is_off_by_default(self) -> None:
        """The deterministic pass is the designed floor.

        ``retrieve.use_query_llm`` is false: the rewrite adds seconds to a
        cascade that is otherwise ~90ms, and the fields that matter come from
        the lexicon regardless.
        """
        parsed = understand("TQ FC7")
        assert parsed.source == "deterministic"


def test_search_text_falls_back_to_the_original_query() -> None:
    """With no rewrite, both channels still get something to search."""
    assert Understanding(query="TQ FC7").search_text == "TQ FC7"
