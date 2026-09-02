"""Tests for settings and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeley_rag import paths
from seeley_rag.exceptions import ConfigurationError
from seeley_rag.settings import REPO_ROOT, Settings, get_models_lexicon, get_settings


class TestSettingsLoading:
    """Loading config/config.yaml."""

    def test_loads_the_real_config(self) -> None:
        """The committed config must actually parse and validate."""
        settings = Settings.from_yaml()
        assert settings.crawl.base_url.startswith("https://")
        assert settings.crawl.rps > 0
        assert settings.pilot_categories

    def test_missing_config_raises_actionable_error(self, tmp_path: Path) -> None:
        """A missing config names what it looked for."""
        with pytest.raises(ConfigurationError, match="not found"):
            Settings.from_yaml(tmp_path / "nope.yaml")

    def test_non_mapping_config_is_rejected(self, tmp_path: Path) -> None:
        """A YAML list where a mapping belongs fails loudly."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            Settings.from_yaml(bad)

    def test_malformed_yaml_is_rejected(self, tmp_path: Path) -> None:
        """A syntax error is reported as a configuration problem."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("a: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            Settings.from_yaml(bad)

    def test_get_settings_is_cached(self) -> None:
        """Repeated imports must not re-read YAML."""
        assert get_settings() is get_settings()


class TestCrawlConfig:
    """Crawl etiquette settings."""

    def test_delay_is_the_inverse_of_rps(self) -> None:
        """1 rps means one second between requests."""
        settings = Settings.from_yaml()
        assert settings.crawl.delay_seconds == pytest.approx(1.0 / settings.crawl.rps)

    def test_user_agent_is_honest(self) -> None:
        """The crawl names itself and gives Seeley a contact address."""
        agent = Settings.from_yaml().crawl.user_agent("0.1.0")
        assert "SeeleyInstallerBot" in agent
        assert "@" in agent

    def test_rps_is_capped(self) -> None:
        """Crawl politeness is a constraint, not a preference.

        Being blocked ends the project, so an unreasonable rate is rejected at
        configuration time rather than discovered at runtime.
        """
        with pytest.raises(ValueError):
            Settings(crawl={"rps": 100.0})  # type: ignore[arg-type]

    def test_base_url_trailing_slash_is_normalised(self) -> None:
        """Path joins must never produce a double slash."""
        settings = Settings(crawl={"base_url": "https://x/"})  # type: ignore[arg-type]
        assert settings.crawl.base_url == "https://x"

    def test_required_paths_cover_articles_and_attachments(self) -> None:
        """The gate must check what the crawl actually fetches.

        Checking only /support/solutions would miss a robots.txt that permits
        browsing but forbids the attachment endpoint the manuals live behind.
        """
        required = Settings.from_yaml().crawl.required_paths
        assert any("solutions" in p for p in required)
        assert any("attachments" in p for p in required)


class TestArticleConfig:
    """Stub-classification settings."""

    def test_stub_threshold_matches_the_build_plan(self) -> None:
        """200 characters, as specified."""
        assert Settings.from_yaml().articles.stub_max_body_chars == 200

    def test_boilerplate_markers_are_configured(self) -> None:
        """Without these, every stub is misclassified as a content article."""
        markers = Settings.from_yaml().articles.boilerplate_markers
        assert markers
        assert any("Safety Notice" in m.start for m in markers)


class TestPaths:
    """The single source of truth for data locations."""

    def test_every_stage_directory_is_under_the_data_root(self) -> None:
        """Nothing escapes data/, so `clean` and .gitignore stay reliable."""
        for directory in paths.ALL_DIRS:
            assert paths.DATA_ROOT in directory.parents or directory == paths.DATA_ROOT

    def test_raw_is_never_in_the_derived_list(self) -> None:
        """`make clean` must be structurally incapable of deleting raw data."""
        assert paths.RAW_DIR not in paths.DERIVED_DIRS
        for directory in paths.DERIVED_DIRS:
            assert paths.RAW_DIR not in directory.parents

    def test_pdf_path_is_content_addressed(self) -> None:
        """The filename is the hash, which is what makes dedupe free."""
        assert paths.pdf_path("abc123").name == "abc123.pdf"
        assert paths.pdf_path("abc123").parent == paths.RAW_PDF_DIR

    def test_html_cache_path_is_stable_and_url_keyed(self) -> None:
        """The same URL always maps to the same cache file."""
        first = paths.html_cache_path("https://x/a")
        second = paths.html_cache_path("https://x/a")
        third = paths.html_cache_path("https://x/b")
        assert first == second
        assert first != third
        assert first.suffix == ".html"

    def test_relative_to_root_is_posix_and_portable(self) -> None:
        """Manifests move between machines and between Windows and Linux."""
        rendered = paths.relative_to_root(REPO_ROOT / "data" / "00_raw" / "x.pdf")
        assert rendered == "data/00_raw/x.pdf"
        assert "\\" not in rendered

    def test_relative_to_root_falls_back_for_outside_paths(self, tmp_path: Path) -> None:
        """A path outside the repo is returned as-is rather than raising."""
        rendered = paths.relative_to_root(tmp_path / "x.pdf")
        assert rendered.endswith("x.pdf")

    def test_ensure_dirs_is_idempotent(self, temp_data_root: Path) -> None:
        """`make init` may be run any number of times."""
        paths.ensure_dirs()
        paths.ensure_dirs()
        assert (temp_data_root / "00_raw" / "pdf").is_dir()
        assert (temp_data_root / "reports").is_dir()

    def test_clean_derived_never_touches_raw(self, temp_data_root: Path) -> None:
        """Re-acquiring costs a 25-minute polite crawl; raw data is the provenance root."""
        raw_file = temp_data_root / "00_raw" / "pdf" / "keep.pdf"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_bytes(b"precious")
        derived = temp_data_root / "02_processed" / "chunks.jsonl"
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_text("derived", encoding="utf-8")

        paths.clean_derived()

        assert raw_file.exists()
        assert raw_file.read_bytes() == b"precious"
        assert not derived.exists()
        assert (temp_data_root / "02_processed").is_dir()


def test_models_lexicon_loads() -> None:
    """The product lexicon is what prevents cross-product contamination."""
    lexicon = get_models_lexicon()
    assert "families" in lexicon
    assert "DGH" in lexicon["families"]
    assert "TQ" in lexicon["families"]["DGH"]["model_codes"]
    assert "fault_code_patterns" in lexicon
