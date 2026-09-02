"""Project exception hierarchy.

Every module raises from this hierarchy rather than built-in exceptions, so a
caller can distinguish "the crawl is blocked" from "the disk is full" without
inspecting messages. Bare ``except:`` is forbidden project-wide; catch one of
these instead.
"""

from __future__ import annotations


class SeeleyRagError(Exception):
    """Base class for every error raised by this project."""


class ConfigurationError(SeeleyRagError):
    """Settings or config files are missing, malformed, or internally inconsistent."""


class AcquisitionError(SeeleyRagError):
    """Stage 1 failed to fetch, parse, or store something from the portal."""


class RobotsDisallowedError(AcquisitionError):
    """The portal's robots.txt forbids a path the crawl requires.

    This is a project gate, not a warning. With no Freshdesk API key available,
    the public crawl is the only acquisition path (build-plan section 3.0), so
    this exception means acquisition is dead until a human resolves it with
    Seeley -- by obtaining an API key, a bulk PDF export, or written permission
    to crawl. No amount of engineering works around it.
    """


class RateLimitedError(AcquisitionError):
    """The portal returned HTTP 429 or 403.

    Raised immediately and never retried. With no API key there is no fallback
    channel, so retrying into a block risks ending the project (build-plan
    section 3.2, rule 5). Escalate to a human.
    """


class ManifestError(SeeleyRagError):
    """The acquisition manifest is missing, malformed, or fails validation."""


class ParseError(SeeleyRagError):
    """Stage 2 failed to extract content from a PDF or HTML document."""
