"""Conservative, auditable eligibility rules for the research map."""

from __future__ import annotations

import re
from dataclasses import dataclass

QUALITY_ASSESSMENT_VERSION = "research-map-eligibility-v1"


def _patterns(*expressions: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(expression, re.IGNORECASE) for expression in expressions)


_NAVIGATION_PATTERNS = _patterns(
    r"click here to(?: view)?(?: linked)? references",
    r"search using [a-z0-9]+ syntax",
    r"this lecture is being recorded",
    r"open access media",
    r"(?:how to )?citing? articles in ieee spectrum",
    r"content continued from (?:the )?front cover",
    r"top \d+ documents accessed(?::.*)?",
)

_CONFERENCE_FRONT_MATTER_PATTERNS = _patterns(
    r"panel sub-committee",
    r"(?:steering|program|technical program|organization|international programming)"
    r" committees?(?:\s+.*)?",
    r"[\w&'’.-]+(?:\s+\d{4})?\s+subcommittees?",
    r"[\w&'’.-]*\d{2,4}\s+committees?",
    r".*[-—]committees",
    r".*\bprogramme committee",
    r"house science committee us congress",
    r"(?:[\w&'’.-]+(?:\s+\d{4})?\s+){0,4}"
    r"(?:(?:technical|research(?: and tutorial)?|paper selection|executive|"
    r"international|scientific|regional|program|organizing|steering|advisory|"
    r"award|administrative|external review)\s+)+(?:sub-?)?committees?"
    r"(?:\s+(?:members|chairs?))?(?:,?\s+continued)?",
    r"(?:program|general|executive|workshop|technical program|publications?|"
    r"publicity/publications) (?:co-?)?chairs?",
    r"(?:organizing|program) committee (?:for|of) [\w&'’.-]+(?:\s+\d{4})?",
    r"[\w&'’.-]+(?:\s+\d{4})? workshop organization",
    r"[\w&'’.-]+\s+\d{4} organization",
    r"organizing committee: technical program committee chairs",
    r"(?:message|welcome message) from (?:the )?.*(?:chairs?|chairmen|president)",
    r".*chairs?' welcome",
    r"(?:ds[nr]|dsn \d{4}) organizers",
    r"workshop organisers",
    r"(?:international|clinical continuing education) (?:program )?chairs",
    r"(?:steering|advisory|executive|regional) (?:board )?members",
    r"(?:scientific|international) advisory board",
    r"(?:research track )?program committee members",
    r"(?:technical )?program committees?",
    r"(?:conference|\d+(?:st|nd|rd|th) imr conference) organization",
)

_PUBLICATION_FRONT_MATTER_PATTERNS = _patterns(
    r"(?:guest )?editorial(?:\b.*)?",
    r".*\beditorial",
    r".*\beditorial board(?: members)?(?:\b.*)?",
    r"new year message from the editor-in-chief",
    r"introduction to editorial board members.*",
    r"acknowledg(?:e)?ments?(?: to| of)? (?:guest )?(?:reviewers|referees)"
    r"(?:\b.*)?",
    r"(?:additional )?reviewers(?:\b.*)?",
    r"(?:jmems )?reviewers[—-]?\d{4}",
    r"\d+ \d{4} reviewers list",
    r".*reviewers with distinction award",
    r"(?:publication )?submission form",
    r"(?:research )?submissions",
    r"submission guidelines",
    r"procedure for submitting a contributed article(?:\b.*)?",
    r"manuscript title",
    r"(?:\d{4}\s+)?annual index(?:\b.*)?",
    r"\d{4}\s+index\b.*",
    r".*\bindex continued(?:\b.*)?",
    r"(?:cumulative )?(?:author|subject|country) index(?:\b.*)?",
    r"ods \d{4} author index",
    r"list of contributors(?:\b.*)?",
    r"table of contents",
    r"contents? continued from .*",
    r"volume \d+.*issue \d+.*",
    r"about the article",
    r"journal homepage(?::.*)?",
    r"transactions/journals department",
    r"(?:selected |\d+\. )?publications(?: list)?(?: stage.*)?",
    r"journal papers",
    r"published books",
    r"reference papers",
    r"additional reading",
    r"author contributions",
    r"short synopsis of the papers",
    r"(?:opening remarks|welcome)(?: and best paper awards)?",
    r"comments and corrections",
    r"an international research journal",
    r"the international arab journal of information technology.*",
    r"new title from .* publishing",
    r"founding editor .*",
    r"editors?,.*",
    r"apa copyright notice:.*",
)

_EVENT_OR_SESSION_PATTERNS = _patterns(
    r"symposium [a-z0-9]+",
    r"keynotes?",
    r"keynote (?:talk|presentation)(?: [ivx]+)?",
    r"poster abstract",
    r"presentation abstract",
    r"workshop summary",
    r"session details:.*",
    r"session \d+:.*",
    r"(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?)"
    r"[,.]? .*room.*",
    r"past lectures",
    r"future issues",
    r"research briefing",
    r"demo hour",
    r"paper selection committee",
    r"[ivx]+ annual sem awards",
    r"\d+ awards",
    r"google faculty research awards \d{4}",
    r"award committee",
)

_CONTAINER_PATTERNS = _patterns(
    r"(?:the )?institute of electrical and electronics engineers(?:,? inc\.?)?"
    r"(?: officers)?",
    r"ieee [\w&'’.,() /-]+(?:society|magazine|transactions|reviews|journal)"
    r"(?:\b.*)?",
    r"at ieee spectrum online",
    r"ieee-usa",
    r"journal of [\w&'’.,() /-]+",
    r"college of engineering materials, devices, circuits",
    r"(?:proceedings of|proc of the) .*",
)

_AFFILIATION_PATTERNS = _patterns(
    r"(?:carnegie mellon university,\s*)?department of [\w&'’.,() /-]+",
    r"[\"']?school of computer science",
    r"(?:.*\|\s*)?dept\. of [\w&'’.,() /-]+",
)

_PLACEHOLDER_PATTERNS = _patterns(
    r"bài-?\d+-research paper",
    r"ad-a\d+[ -]\d+",
    r"final research report",
    r"reason for research",
)

_EVENT_CODE_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z0-9&.-]{1,15}|[A-Z][a-z]+[A-Z][A-Za-z]*)"
    r"[ :]?(?:19|20)\d{2}$"
)
_ADDRESS_PATTERN = re.compile(
    r"(?:\b\d{5}(?:-\d{4})?\b|\b(?:street|st\.|road|rd\.|avenue|ave\.|"
    r"lane|ln\.|po box)\b|\b[a-z0-9._%+-]+\s*@\s*[a-z0-9.-]+\s*\.\s*"
    r"[a-z]{2,}\b)",
    re.IGNORECASE,
)
_INSTITUTION_PATTERN = re.compile(
    r"\b(?:university|department|institute|center|centre|laborator|editorial)\b",
    re.IGNORECASE,
)
_BIOGRAPHY_PATTERN = re.compile(
    r"\b(?:is an? (?:associate |assistant |full )?professor|"
    r"research interests include)\b",
    re.IGNORECASE,
)
_CMU_DOCUMENT_ID_PATTERN = re.compile(
    r"\bCMU-(?:ISR|CS|ML)-\d{2}-\d{3}\b",
    re.IGNORECASE,
)
_COMMA_NAME_PATTERN = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+,\s*"
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.]+"
)
_INSTITUTION_NAME_PATTERN = re.compile(
    r"\b(?:university|institute|google|microsoft|nvidia|ibm|inc\.)\b",
    re.IGNORECASE,
)
_REPEATED_CAPITAL_PATTERN = re.compile(r"[A-Z]{20,}")
_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+")
_AUTHOR_INITIAL_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,2}|[A-Z]\.)\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’-]{2,}"
)
_EDITOR_SUFFIX_PATTERN = re.compile(r"\(editors?\)$", re.IGNORECASE)
_NAME_CONNECTORS = frozenset(
    {"and", "de", "del", "der", "di", "du", "la", "le", "of", "the", "van", "von"}
)


@dataclass(frozen=True, slots=True)
class PublicationQuality:
    """One deterministic map-eligibility decision and its evidence."""

    map_eligible: bool
    exclusion_reasons: tuple[str, ...]


def _matches(patterns: tuple[re.Pattern[str], ...], title: str) -> bool:
    return any(pattern.fullmatch(title) for pattern in patterns)


def _has_garbled_index_text(title: str) -> bool:
    if "....." in title:
        return True
    return any(
        len(run) >= 20 and len(set(run)) <= 5
        for run in _REPEATED_CAPITAL_PATTERN.findall(title)
    )


def _looks_like_person_or_citation_index(title: str, *, metadata_weak: bool) -> bool:
    if len(_CMU_DOCUMENT_ID_PATTERN.findall(title)) >= 2:
        return True
    comma_names = _COMMA_NAME_PATTERN.findall(title)
    starts_with_name = _COMMA_NAME_PATTERN.match(title) is not None
    page_references = re.findall(r"(?<![A-Za-z])\d{1,4}(?![A-Za-z])", title)
    if starts_with_name and len(comma_names) >= 2 and len(page_references) >= 3:
        return True
    institutions = _INSTITUTION_NAME_PATTERN.findall(title)
    if (
        metadata_weak
        and len(institutions) >= 2
        and (title.count(",") >= 2 or title.count("(") >= 2)
    ):
        return True

    words = _WORD_PATTERN.findall(title)
    name_words = [word for word in words if word.casefold() not in _NAME_CONNECTORS]
    capitalized = sum(bool(re.match(r"^[A-ZÀ-ÖØ-Þ]", word)) for word in name_words)
    capitalized_ratio = capitalized / max(len(name_words), 1)
    initial_names = len(_AUTHOR_INITIAL_PATTERN.findall(title))
    strong_initial_list = initial_names >= 3 and len(words) <= 18
    editor_list = _EDITOR_SUFFIX_PATTERN.search(title) is not None
    weak_name_markers = (
        ";" in title
        or "**" in title
        or re.search(r"\b[A-Z]\.", title)
        or initial_names >= 2
        or editor_list
    )
    comma_list = title.count(",") >= 3 and ":" not in title and "?" not in title
    return bool(
        capitalized_ratio >= 0.95
        and (
            strong_initial_list
            or editor_list
            or (
                metadata_weak
                and 2 <= len(words) <= 24
                and (weak_name_markers or comma_list)
            )
        )
    )


def assess_publication(
    *,
    title: str,
    year: int | None,
    doi: str,
    source_url: str,
) -> PublicationQuality:
    """Exclude only high-confidence non-research records from map computation."""

    clean_title = " ".join(str(title or "").split())
    metadata_weak = year is None and not doi.strip() and not source_url.strip()
    reasons: list[str] = []

    if _matches(_NAVIGATION_PATTERNS, clean_title):
        reasons.append("navigation_or_interface")
    if _matches(_CONFERENCE_FRONT_MATTER_PATTERNS, clean_title):
        reasons.append("conference_front_matter")
    if _matches(_PUBLICATION_FRONT_MATTER_PATTERNS, clean_title):
        reasons.append("publication_front_matter")
    if (
        _matches(_EVENT_OR_SESSION_PATTERNS, clean_title)
        or _EVENT_CODE_PATTERN.fullmatch(clean_title)
    ) and (metadata_weak or len(clean_title.split()) <= 4):
        reasons.append("event_or_session_label")
    if _matches(_CONTAINER_PATTERNS, clean_title) and (
        metadata_weak or len(clean_title.split()) <= 10
    ):
        reasons.append("organization_or_container")

    affiliation = metadata_weak and _matches(_AFFILIATION_PATTERNS, clean_title)
    contact = _ADDRESS_PATTERN.search(clean_title) and (
        _INSTITUTION_PATTERN.search(clean_title) or "@" in clean_title
    )
    if affiliation or contact or _BIOGRAPHY_PATTERN.search(clean_title):
        reasons.append("affiliation_or_contact")
    if _matches(_PLACEHOLDER_PATTERNS, clean_title):
        reasons.append("placeholder_or_document_control")
    if _looks_like_person_or_citation_index(
        clean_title,
        metadata_weak=metadata_weak,
    ):
        reasons.append("person_or_citation_index")
    if _has_garbled_index_text(clean_title):
        reasons.append("garbled_or_index_text")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return PublicationQuality(
        map_eligible=not unique_reasons,
        exclusion_reasons=unique_reasons,
    )
