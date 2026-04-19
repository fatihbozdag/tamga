"""Language registry — one first-class entry per supported language."""

from tamga.languages.registry import REGISTRY as LANGUAGES
from tamga.languages.registry import LanguageSpec
from tamga.languages.registry import get as get_language

__all__ = ["LANGUAGES", "LanguageSpec", "get_language"]
