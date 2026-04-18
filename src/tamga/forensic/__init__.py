"""Forensic-linguistics toolkit — authorship verification, LR-framed reporting, and related
methods aimed at forensic research and evidentiary-standard output.

Every method here is designed around forensic assumptions: small known/questioned document
sets, open-set rather than closed-set decisions, and reporting suitable for journals such as
*International Journal of Speech, Language and the Law*, *Language and Law*, and *Applied
Linguistics*.
"""

from tamga.forensic.verify import GeneralImpostors

__all__ = ["GeneralImpostors"]
