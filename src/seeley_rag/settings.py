"""Application settings: ``config/*.yaml`` layered under ``.env``.

Resolution order, lowest precedence first:

1. Defaults declared on the pydantic models below.
2. ``config/config.yaml`` -- non-secret, committed, the normal place to change
   a value.
3. Environment variables prefixed ``SEELEY_`` and ``.env`` -- secrets and
   per-machine overrides. Only a curated few are wired; see :class:`Settings`.

Secrets never appear in YAML. Non-secrets never appear in ``.env``. That split
is what lets ``config/`` be committed while ``.env`` stays ignored.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seeley_rag.exceptions import ConfigurationError

# This file is src/seeley_rag/settings.py, so the repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = REPO_ROOT / "config"
CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"
MODELS_FILE: Path = CONFIG_DIR / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk.

    Args:
        path: File to read.

    Returns:
        The parsed mapping, or an empty dict if the file is empty.

    Raises:
        ConfigurationError: If the file is missing or is not a mapping.
    """
    if not path.exists():
        raise ConfigurationError(
            f"Config file not found: {path}. Expected it under the repository root's "
            "config/ directory."
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level.")
    return loaded


class ProjectConfig(BaseModel):
    """Project identity, stamped into every artefact for provenance."""

    name: str = "seeley-rag"
    crawler_version: str = "0.1.0"


class CrawlConfig(BaseModel):
    """Crawl etiquette and portal endpoints.

    These are not tuning knobs. With no API key the public crawl is the only
    acquisition path, so being blocked ends the project (build-plan section 3.0).
    """

    base_url: str = "https://seeleyinternationalhelp.freshdesk.com"
    solutions_path: str = "/support/solutions"
    rps: float = Field(default=1.0, gt=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=5)
    retry_backoff_seconds: float = Field(default=2.0, gt=0.0)
    max_pages_per_folder: int = Field(default=50, ge=1)
    user_agent_template: str = "SeeleyInstallerBot/{version} (+{contact})"
    contact: str = "shlok@rostered.ai"
    required_paths: list[str] = Field(
        default_factory=lambda: [
            "/support/solutions",
            "/support/solutions/articles",
            "/helpdesk/attachments",
        ]
    )

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalise the base URL so path joins never produce a double slash."""
        return value.rstrip("/")

    @property
    def delay_seconds(self) -> float:
        """Seconds to sleep between requests in order to honour :attr:`rps`."""
        return 1.0 / self.rps

    def user_agent(self, version: str = "0.1.0") -> str:
        """Build the honest User-Agent string, contact address included.

        Args:
            version: Crawler version to embed.

        Returns:
            A User-Agent naming the bot, its version, and a reachable contact.
        """
        return self.user_agent_template.format(version=version, contact=self.contact)


class BoilerplateMarker(BaseModel):
    """A span of shared boilerplate to remove from article bodies.

    Attributes:
        start: Substring that opens the block.
        end: Substring that closes it. The span removed runs from ``start`` up
            to and including ``end``.
    """

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class ArticleConfig(BaseModel):
    """Thresholds for the acquisition-time stub/content split.

    build-plan section 4.4: a body shorter than ``stub_max_body_chars`` that
    also has an attachment is a card-catalogue entry. Indexing its text would
    pollute retrieval with "Pdf attached" chunks.

    ``boilerplate_markers`` exists because the portal changed after the build
    plan was written: a 1026-character safety notice is now appended to article
    bodies, identical across articles. It has to be stripped before the
    threshold above can mean anything. See ADR 0004.
    """

    stub_max_body_chars: int = Field(default=200, ge=1)
    boilerplate_markers: list[BoilerplateMarker] = Field(default_factory=list)


class TriageConfig(BaseModel):
    """Per-page classification thresholds for the Stage 0 PDF triage.

    build-plan section 4.1. The three fractions this produces set the vision
    budget for the entire project.
    """

    text_layer_min_chars: int = Field(default=100, ge=0)
    diagram_heavy_max_chars: int = Field(default=600, ge=0)
    diagram_heavy_min_images: int = Field(default=1, ge=0)


class ParseConfig(BaseModel):
    """Stage 2 parsing configuration.

    build-plan section 4.3: every page is rendered, at 150 DPI, giving roughly
    150-250 KB per page. Across this corpus's 12,526 pages that is about 2-3 GB
    -- fine locally, object storage in production.
    """

    render_dpi: int = Field(default=150, ge=36, le=600)


class ChunkConfig(BaseModel):
    """Stage 3 chunking sizes. build-plan section 5.1.

    ``table_max_tokens`` is not a tuning knob but a hard external constraint:
    ``text-embedding-3-large`` rejects inputs above 8,191 tokens and Cohere
    rerank truncates near 4,000. A table chunk above the cap fails to embed --
    on exactly the fault-code content the system exists to serve.
    """

    target_tokens: int = Field(default=800, ge=64)
    max_tokens: int = Field(default=1200, ge=64)
    overlap_tokens: int = Field(default=120, ge=0)
    table_max_tokens: int = Field(default=6000, ge=256, le=8000)
    #: Bodies shorter than this are dropped rather than embedded. 897 pages in
    #: the corpus are near-empty scan artefacts; indexing them would put
    #: citable-but-contentless rows in the store.
    min_chunk_chars: int = Field(default=50, ge=0)

    @field_validator("max_tokens")
    @classmethod
    def _max_above_target(cls, value: int, info: Any) -> int:
        """Reject a hard cap below the target, which would make every chunk oversized."""
        target = info.data.get("target_tokens")
        if target is not None and value < target:
            raise ValueError(f"max_tokens ({value}) must be >= target_tokens ({target}).")
        return value


class IndexConfig(BaseModel):
    """Stage 4 embedding and vector-store configuration. build-plan section 6.

    ``embedding_dim`` stays at the model's native 3,072. OpenAI supports
    truncating to fewer dimensions at the same price, which shrinks the index
    but costs measurable recall; accuracy is the binding constraint here, so
    the trade is left as a config change the eval can measure rather than an
    unmeasured default.
    """

    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = Field(default=3072, ge=256, le=3072)
    batch_size: int = Field(default=256, ge=1, le=2048)
    table_name: str = "chunks"


class RetrieveConfig(BaseModel):
    """Stage 5 retrieval configuration. build-plan section 7.2.

    Every boost here is a multiplier on the *fused* score, applied before the
    list is truncated for reranking. That ordering is the point: a boost applied
    to an already-truncated top-8 can only reorder what survived, never promote
    the chunk that should have been there.

    None of them is a filter. Retrieval soft-boosts an inferred product family
    because a confident wrong guess plus a hard filter returns nothing and makes
    the system look broken (section 7.1).
    """

    # `model_series_boost` is a real domain field -- Seeley model codes -- not a
    # pydantic accessor, so the protected namespace is cleared as it is on the
    # Page and Chunk models.
    model_config = ConfigDict(protected_namespaces=())

    dense_top_k: int = Field(default=30, ge=1)
    bm25_top_k: int = Field(default=30, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=8, ge=1)
    diagnostic_article_boost: float = Field(default=1.2, ge=1.0)
    product_family_boost: float = Field(default=1.35, ge=1.0)
    model_series_boost: float = Field(default=1.15, ge=1.0)
    code_match_boost: float = Field(default=1.5, ge=1.0)
    max_pinned_codes: int = Field(default=3, ge=0)

    #: Let the router rewrite the query before searching. Off by default: the
    #: deterministic pass already supplies family, models and codes, and the
    #: rewrite adds ~1-4s to a cascade that is otherwise ~90ms.
    use_query_llm: bool = False

    #: Listwise LLM reranking when no Cohere key exists. Off by default because
    #: build-plan section 7.2 warns it roughly doubles per-query cost. Cohere is
    #: preferred where a key is available; this is the plan's own fallback.
    use_llm_rerank: bool = False

    #: Candidates sent to the listwise reranker. Beyond this the prompt grows
    #: faster than the ranking improves.
    llm_rerank_candidates: int = Field(default=20, ge=1, le=50)

    #: Model for the listwise reranker. ``None`` inherits ``generate.router_model``,
    #: which is what this did implicitly before the field existed -- and that is a
    #: silent mismatch worth being able to correct: the router model is chosen for
    #: latency (build-plan section 7.1), while reranking is a judgement task where
    #: the plan expects a purpose-built cross-encoder. Naming it makes the choice
    #: reviewable and lets an eval attribute a rerank number to a model.
    llm_rerank_model: str | None = None


class GenerateConfig(BaseModel):
    """Model choices for Stage 6 generation and the Stage 5 query router.

    build-plan sections 7.1 and 8. Declared here rather than in the generation
    stage because Stage 5's query-understanding pass reaches for the router
    model, and one place to change a model name is worth more than tidy layering.

    The plan names Claude. Nothing in the architecture requires it, and this
    project has an OpenAI key covering generation, routing and the outstanding
    vision work, so ``openai`` is the default. Switching back is two lines here
    plus an ``ANTHROPIC_API_KEY``; see ADR 0008.
    """

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5"
    router_model: str = "gpt-4.1-mini"
    #: Reasoning models cost seconds a router cannot spend -- gpt-5-mini takes
    #: 8.3s by default and 2.1s at "minimal". Ignored by non-reasoning models.
    reasoning_effort: str = "minimal"


class LoggingConfig(BaseModel):
    """Structured-logging configuration."""

    level: str = "INFO"
    format: str = "json"

    @field_validator("level")
    @classmethod
    def _upper(cls, value: str) -> str:
        """Accept lower-case level names from YAML or the environment."""
        return value.upper()


class Settings(BaseSettings):
    """Top-level settings object. Construct via :func:`get_settings`.

    Attributes:
        data_root: Root of the local data tree. Every path in the project is
            derived from this by ``paths.py``; no directory string literals live
            anywhere else.
        crawler_contact: Contact address embedded in the crawl User-Agent.
        openai_api_key: Stage 4 embeddings. Unused while downstream is stubbed.
        anthropic_api_key: Stage 2b vision and Stage 6 generation. Unused yet.
        cohere_api_key: Stage 5 reranking. Unused yet.
        freshdesk_api_key: Currently unobtainable -- ``/api/v2/*`` returns 401.
            Kept so an ``ApiClient`` implementation can appear later without a
            settings change (see ADR 0002).
    """

    model_config = SettingsConfigDict(
        env_prefix="SEELEY_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- Overridable via SEELEY_* env vars --------------------------------
    data_root: Path = Path("data")
    log_level: str | None = None
    crawl_rps: float | None = None
    crawler_contact: str | None = None

    # --- Secrets: read from .env without the SEELEY_ prefix ---------------
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    freshdesk_api_key: str | None = Field(default=None, alias="FRESHDESK_API_KEY")
    page_image_base_url: str | None = Field(default=None, alias="PAGE_IMAGE_BASE_URL")

    # --- Loaded from config/config.yaml -----------------------------------
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    articles: ArticleConfig = Field(default_factory=ArticleConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pilot_categories: list[str] = Field(default_factory=list)

    @field_validator(
        "log_level",
        "crawl_rps",
        "crawler_contact",
        "openai_api_key",
        "anthropic_api_key",
        "cohere_api_key",
        "freshdesk_api_key",
        "page_image_base_url",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        """Treat an empty environment variable as absent.

        ``.env.example`` ships every optional override as a bare ``NAME=``, so
        copying it verbatim -- which is exactly what the README tells you to do
        -- would otherwise fail settings validation with "unable to parse string
        as a number". An empty value means "not set", not "set to nothing".

        Args:
            value: The raw value from the environment or YAML.

        Returns:
            ``None`` for blank strings, the value otherwise.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @classmethod
    def from_yaml(cls, config_file: Path | None = None) -> Settings:
        """Build settings from ``config/config.yaml``, then apply env overrides.

        Args:
            config_file: Alternative config path. Defaults to
                ``config/config.yaml`` at the repository root.

        Returns:
            A fully resolved settings object.

        Raises:
            ConfigurationError: If the config file is missing or malformed.
        """
        raw = _load_yaml(config_file or CONFIG_FILE)
        payload: dict[str, Any] = {
            "project": raw.get("project", {}),
            "crawl": raw.get("crawl", {}),
            "articles": raw.get("articles", {}),
            "triage": raw.get("triage", {}),
            "parse": raw.get("parse", {}),
            "chunk": raw.get("chunk", {}),
            "index": raw.get("index", {}),
            "retrieve": raw.get("retrieve", {}),
            "generate": raw.get("generate", {}),
            "logging": raw.get("logging", {}),
            "pilot_categories": raw.get("pilot_categories", []),
        }
        data_root = raw.get("paths", {}).get("data_root")
        if data_root:
            payload["data_root"] = Path(data_root)

        settings = cls(**payload)

        # Env vars win over YAML for the handful of fields that allow it.
        if settings.log_level:
            settings.logging.level = settings.log_level.upper()
        if settings.crawl_rps:
            settings.crawl.rps = settings.crawl_rps
        if settings.crawler_contact:
            settings.crawl.contact = settings.crawler_contact
        return settings

    @property
    def resolved_data_root(self) -> Path:
        """Absolute path to the data tree.

        Relative values resolve against the repository root, so the data tree's
        location never depends on the directory a script was launched from.
        """
        if self.data_root.is_absolute():
            return self.data_root
        return REPO_ROOT / self.data_root


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so repeated imports do not re-read YAML. Call
    ``get_settings.cache_clear()`` in tests that need a fresh read.

    Returns:
        The resolved :class:`Settings`.
    """
    return Settings.from_yaml()


@functools.lru_cache(maxsize=1)
def get_models_lexicon() -> dict[str, Any]:
    """Return the product-family / model-code lexicon from ``config/models.yaml``.

    Nothing in Stage 1 consumes this -- the manifest records category and folder
    verbatim. It is loaded here so Stage 3 has a single accessor to reach for.

    Returns:
        The parsed lexicon mapping.

    Raises:
        ConfigurationError: If the file is missing or malformed.
    """
    return _load_yaml(MODELS_FILE)
