"""Language registry — one first-class entry per supported language."""

from bitig.languages.registry import REGISTRY as LANGUAGES
from bitig.languages.registry import LanguageSpec
from bitig.languages.registry import get as get_language

__all__ = ["LANGUAGES", "LanguageSpec", "get_language"]
