"""Stage 6 prompt construction.

build-plan section 8. These tests assert that the safety and grounding rules are
actually present in what gets sent, and that context assembly does not lose or
misattribute a passage.

A rule that quietly disappears from the prompt is invisible: answers keep
arriving, they are just less grounded. So each requirement §8 lists is asserted
by name.
"""

from __future__ import annotations

from typing import Any

from seeley_rag.chunk.base import FaultCode
from seeley_rag.generate.prompts import (
    MAX_PASSAGE_CHARS,
    build_context,
    build_user_message,
    render_passage,
    render_pinned_code,
    system_prompt,
)
from seeley_rag.retrieve.hybrid import PinnedCode


def chunk(**overrides: Any) -> dict[str, Any]:
    """Build a retrieved chunk row.

    Args:
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, Any] = {
        "chunk_id": "c1",
        "text": "Set the flame sensor gap to 4-6 mm.",
        "title": "TQ Service Guide",
        "page_label": "42",
        "product_family": "DGH",
        "kind": "prose",
    }
    base.update(overrides)
    return base


class TestSystemPromptRequirements:
    """Every rule build-plan section 8 lists, asserted by name."""

    def test_answers_only_from_context(self) -> None:
        """The defining constraint of the whole system."""
        prompt = system_prompt().lower()
        assert "only from those passages" in prompt

    def test_requires_inline_citations(self) -> None:
        """Citations are what make an answer checkable."""
        assert "[1]" in system_prompt()
        assert "inline citation" in system_prompt().lower()

    def test_requires_saying_when_the_answer_is_absent(self) -> None:
        """A fluent ungrounded answer is indistinguishable from a real one."""
        prompt = system_prompt().lower()
        assert "do not answer from general" in prompt

    def test_forbids_rounding_exact_values(self) -> None:
        """Gas pressures and torque figures are the reason this system exists."""
        prompt = system_prompt().lower()
        assert "never round" in prompt
        assert "gas pressures" in prompt

    def test_requires_licensed_technician_warning(self) -> None:
        """Gas carriage, combustion and mains electrical work."""
        prompt = system_prompt().lower()
        assert "licensed technician" in prompt
        assert "mains electrical" in prompt

    def test_forbids_synthesising_absent_procedures(self) -> None:
        """A completed procedure that the manual did not give is a safety risk."""
        assert "never describe a procedure step that does not appear" in system_prompt().lower()

    def test_manuals_win_over_the_model_prior(self) -> None:
        """Section 8 states this explicitly, and it is easy to lose in editing."""
        assert "the passages win" in system_prompt().lower()

    def test_forbids_defeating_safety_devices(self) -> None:
        """The most dangerous plausible suggestion in this domain."""
        prompt = system_prompt().lower()
        assert "bypassing, defeating or removing a safety device" in prompt

    def test_specifies_the_field_answer_layout(self) -> None:
        """The UI renders these headings; a drifting prompt silently breaks it."""
        prompt = system_prompt()
        for heading in ("Answer:", "What to check:", "Technician-only work:"):
            assert heading in prompt

    def test_technician_section_is_conditional(self) -> None:
        """Boilerplate on every answer trains installers to skip it."""
        prompt = system_prompt().lower()
        assert "include this section only when" in prompt

    def test_forbids_a_sources_section(self) -> None:
        """Citations are rendered by the application from the [n] markers."""
        prompt = system_prompt().lower()
        assert 'do not write a "sources" section' in prompt

    def test_forbids_tables_and_markup(self) -> None:
        """The answer is read on a phone, as plain text."""
        prompt = system_prompt().lower()
        assert "no markdown tables" in prompt
        assert "no html" in prompt

    def test_requires_enumerating_a_code_with_no_product(self) -> None:
        """Installers type "fc7", not "TQ heater showing FC7"."""
        prompt = system_prompt().lower()
        assert "fault code with no product named is ambiguous" in prompt
        assert "never pick one family" in prompt

    def test_the_prose_attribution_example_is_not_copyable(self) -> None:
        """The old example named a family, and the model copied it verbatim.

        A DGH row came back attributed to "the RC service manual", because the
        prompt offered that phrase as the template rather than as an example.
        """
        prompt = system_prompt()
        assert "naming the source the block itself" in prompt

    def test_asks_for_structured_output(self) -> None:
        """The response fields the API contract needs."""
        prompt = system_prompt()
        for field in ("answer", "confidence", "answered", "missing"):
            assert f'"{field}"' in prompt


class TestPassageRendering:
    """The model must be able to tell two passages apart to cite them."""

    def test_passage_is_numbered(self) -> None:
        """The number is the citation handle."""
        assert render_passage(3, chunk()).startswith("[3] ")

    def test_header_carries_title_and_page(self) -> None:
        """So a page written in prose matches the resolved citation."""
        rendered = render_passage(1, chunk())
        assert "TQ Service Guide" in rendered
        assert "p.42" in rendered

    def test_page_range_is_shown_for_a_merged_table(self) -> None:
        """A merged fault table spans pages and must say so."""
        assert "p.42-44" in render_passage(1, chunk(page_range="42-44"))

    def test_tables_are_labelled(self) -> None:
        """A fault-code table is different evidence from prose about one."""
        assert "table)" in render_passage(1, chunk(kind="table"))

    def test_body_is_bounded(self) -> None:
        """A merged table can be large; the prompt still has to fit."""
        rendered = render_passage(1, chunk(text="x" * 50_000))
        assert len(rendered) < MAX_PASSAGE_CHARS + 500

    def test_a_missing_page_label_is_omitted_not_guessed(self) -> None:
        """630 diagnostic articles have no printed page at all."""
        assert "p.None" not in render_passage(1, chunk(page_label=None))


class TestPinnedCodes:
    """Exact lookups lead the context -- build-plan section 5.3."""

    def pin(self, cross_family: bool = False, ambiguous: bool = False) -> PinnedCode:
        """Build a pinned code row.

        Args:
            cross_family: Whether the code belongs to another family.
            ambiguous: Whether the query named no product at all.

        Returns:
            The pin.
        """
        return PinnedCode(
            row=FaultCode(
                code="E4",
                code_key="E04",
                meaning="High discharge temperature protection",
                product_family="VRF",
                title="VRF Service Manual",
                page_label="13",
            ),
            cross_family=cross_family,
            ambiguous=ambiguous,
        )

    def test_a_matching_code_is_stated_plainly(self) -> None:
        """The straightforward case."""
        rendered = render_pinned_code(self.pin())
        assert "E4" in rendered
        assert "High discharge temperature protection" in rendered
        assert "NOT" not in rendered

    def test_a_cross_family_code_is_flagged_emphatically(self) -> None:
        """The "ducted heater throwing E:04" case.

        DGH prints FC codes, so there is no DGH E04. Presenting the VRF meaning
        as the answer would be a confident wrong answer with a citation.
        """
        rendered = render_pinned_code(self.pin(cross_family=True))
        assert "NOT" in rendered
        assert "do not present it as the answer" in rendered

    def test_an_ambiguous_code_is_labelled_by_family_not_answered(self) -> None:
        """The shape a trade worker types: a code and no product.

        A bare "fc7" is a gas-heater ignition failure AND an evaporative
        supply-motor error. Picking one produced a confident, high-confidence
        answer about the wrong appliance.
        """
        rendered = render_pinned_code(self.pin(ambiguous=True))
        assert "did NOT say which product" in rendered
        assert "more than one product family" in rendered
        assert "labelled by family" in rendered

    def test_an_ambiguous_set_is_not_headed_as_authoritative(self) -> None:
        """Calling a set of contradictory meanings an exact lookup invites a pick."""
        context = build_context([chunk()], [self.pin(ambiguous=True)])
        assert "AMBIGUOUS" in context
        assert "EXACT FAULT-CODE LOOKUP" not in context

    def test_pinned_codes_lead_the_context(self) -> None:
        """They are exact lookups, not retrieval guesses."""
        context = build_context([chunk()], [self.pin()])
        assert context.index("EXACT FAULT-CODE LOOKUP") < context.index("PASSAGES:")


class TestContextAssembly:
    """What actually reaches the model."""

    def test_passages_are_numbered_from_one(self) -> None:
        """Citation numbers are 1-based; an off-by-one misattributes every claim."""
        context = build_context([chunk(title="First"), chunk(title="Second")])
        assert "[1] First" in context
        assert "[2] Second" in context

    def test_empty_retrieval_is_stated_not_hidden(self) -> None:
        """An empty passage list must not look like a passage list."""
        assert "(none retrieved)" in build_context([])

    def test_the_question_leads_the_user_message(self) -> None:
        """The model reads the question before the evidence."""
        message = build_user_message("TQ FC7?", [chunk()])
        assert message.startswith("QUESTION: TQ FC7?")

    def test_context_follows_the_question(self) -> None:
        """Both halves are present and in order."""
        message = build_user_message("TQ FC7?", [chunk()], [])
        assert message.index("QUESTION:") < message.index("PASSAGES:")
