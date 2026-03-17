"""Ingestion module."""

from .normalize import compose_normalized_text
from .parsers import parse_raw_page, parse_raw_pages

__all__ = ["compose_normalized_text", "parse_raw_page", "parse_raw_pages"]
