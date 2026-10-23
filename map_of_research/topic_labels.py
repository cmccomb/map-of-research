"""Human-reviewed display labels for automatically extracted map topics.

The keys are exact phrases produced by the deterministic title-based extractor.
Unknown phrases remain visible as extracted, so a changing corpus never loses a
topic merely because its wording has not yet received editorial review.
"""

from __future__ import annotations

from collections.abc import Mapping

TOPIC_LABEL_REVIEW_VERSION = "representative-title-audit-2026-07-19-v1"

# Every overview phrase in the 2026-07-19 corpus was checked against its child
# regions and representative titles. Broad compound labels are intentional when
# a zoomed-out spatial region spans several neighboring research communities.
REVIEWED_OVERVIEW_LABELS = {
    "access control": "cybersecurity & privacy",
    "autonomous vehicles": "autonomous vehicles",
    "body surface": "biomedical devices & modeling",
    "brain computer interface": "neural engineering & imaging",
    "carbon nanotubes": "nanomaterials, therapeutics & particle physics",
    "climate change": "climate & energy systems",
    "cmos mems": "microsystems & electronic materials",
    "contour mode": "robotics, resonators & photonics",
    "cyber physical": "software, networks & society",
    "decision making": "risk & decision science",
    "electric vehicle": "power, energy & mobility",
    "face recognition": "computer vision & imaging",
    "fault tolerance": "dependable computing & networks",
    "fuel cell": "electrochemistry & surface science",
    "grain boundary": "materials processing & mechanics",
    "growth factor": "cellular & biomolecular engineering",
    "heat exchanger": "process & materials engineering",
    "life cycle assessment": "design, policy & life-cycle analysis",
    "liquid metal": "soft materials & photonics",
    "magnetic properties": "magnetic & functional materials",
    "mesh generation": "computational modeling & circuits",
    "organic aerosol": "atmospheric chemistry",
    "nonlinear predictive control": "optimization & control",
    "real time": "computer systems & architecture",
    "reinforcement learning": "intelligent & resilient systems",
    "sensor networks": "networked intelligence & communications",
    "silicon carbide": "semiconductor & environmental nanomaterials",
    "speech recognition": "speech & signal processing",
    "supply chain": "process systems & supply chains",
    "thermal conductivity": "materials characterization & transport",
}

# Detail phrases were checked directly against representative publication titles.
# The reviewed form fixes truncation, jargon, accidental codes, and labels that
# described only a small, highly repeated subset of their region.
REVIEWED_DETAIL_LABELS = {
    "4h sic": "silicon carbide materials",
    "access control": "binary analysis & security",
    "activated carbon": "chemical modeling & remediation",
    "air pollution": "air pollution & sensing",
    "aluminum nitride": "integrated photonics & acoustics",
    "analog rf": "analog & RF circuits",
    "anti phishing": "anti-phishing",
    "artificial lung": "artificial lungs",
    "automated vehicles": "automated vehicles",
    "batch plants": "production planning & scheduling",
    "block copolymer": "block copolymers",
    "body surface": "cardiac electrical imaging",
    "brain computer interface": "brain-computer interfaces",
    "building information": "construction informatics",
    "carbon nanotubes": "carbon nanotubes",
    "central bank digital currency": "digital currency",
    "circulating tumor cells": "circulating tumor cells",
    "click beetles": "bio-inspired jumping robots",
    "climate change": "climate & energy transitions",
    "cmos mems": "CMOS-MEMS integration",
    "connected vehicle": "connected vehicles",
    "continuous casting": "steelmaking & casting",
    "contour mode": "MEMS resonators",
    "cost benefit": "policy decisions & valuation",
    "cu 111": "surface chemistry",
    "cyber physical": "wireless spectrum sharing",
    "ddos attacks": "DDoS & network security",
    "decision making": "trauma triage",
    "deep reinforcement learning": "machine learning & explainability",
    "design teams": "design cognition & teams",
    "distillation systems": "distillation & reactor design",
    "distinguished scientific early": "design contests & awards",
    "doped diamond like carbon": "magnetic & carbon thin films",
    "electric vehicle": "electric vehicles",
    "electricity markets": "electricity systems & policy",
    "extracellular matrix": "extracellular matrix",
    "face recognition": "face recognition",
    "fault tolerant": "fault-tolerant systems",
    "federated learning": "federated learning",
    "field dislocation mechanics": "dislocation mechanics",
    "flow focusing": "microfluidics & liquid metals",
    "fuel cell": "fuel cells",
    "generalized disjunctive programming": "mixed-integer optimization",
    "grain boundary": "microstructure & grain growth",
    "grain growth": "materials simulation",
    "graph signal processing": "graph signal processing & learning",
    "heat exchanger": "process integration & carbon capture",
    "high mountain asia": "mountain hydrology & climate",
    "high speed": "microsystems & mechanics",
    "hip exoskeleton": "wearable robotics",
    "hyaluronic acid": "drug delivery & wound healing",
    "images fundamentals": "visual navigation & image modeling",
    "information flow": "software & network systems",
    "instruction set": "computer architecture",
    "insulin like growth factor": "insulin-like growth factor",
    "integrated services": "adaptive video & network services",
    "la0 7sr0 3mno3": "functional oxide thin films",
    "laser powder bed fusion": "powder-bed fusion",
    "ldpc codes": "error-correcting codes",
    "life cycle assessment": "life-cycle assessment",
    "liquid metal": "stretchable electronics",
    "lithium ion": "lithium-ion batteries",
    "magnetic properties": "soft magnetic materials",
    "magnetic recording": "magnetic recording",
    "magnetoacoustic tomography magnetic induction": "magnetoacoustic imaging",
    "mesh generation": "finite-element meshing",
    "microneedle arrays": "microneedles & drug delivery",
    "mlc nand flash memory": "flash storage",
    "molecular weight distribution": "process modeling & intensification",
    "multi agent": "autonomous planning & uncertainty",
    "nanoscale zerovalent iron": "zero-valent iron nanoparticles",
    "non intrusive load monitoring": "building energy monitoring",
    "nonlinear predictive control": "nonlinear predictive control",
    "object detection": "object detection",
    "organic aerosol": "secondary organic aerosols",
    "origin destination": "transportation modeling",
    "parallel monitoring": "distributed systems observability",
    "parameter estimation": "nonlinear estimation & control",
    "particle formation": "atmospheric particle formation",
    "pattern search": "large-scale & design optimization",
    "photochemical reactivity": "surface chemistry & photovoltaics",
    "photonic crystals": "photonic crystals",
    "point clouds": "computational imaging",
    "power flow": "power-system optimization",
    "pp collisions": "particle physics",
    "privacy security": "usable privacy & security",
    "process synthesis": "process systems optimization",
    "product family": "product families & platforms",
    "public safety": "internet economics & policy",
    "pulsed laser deposition": "pulsed-laser deposition",
    "ray diffraction": "X-ray diffraction & texture",
    "real time": "data center & real-time networks",
    "real time systems": "real-time systems",
    "reinforcement learning": "multi-agent learning & planning",
    "resistive switching devices": "microscopy & materials characterization",
    "risk communication": "risk communication",
    "saccharomyces cerevisiae": "cellular protein engineering",
    "sensor networks": "random graphs & sensor networks",
    "signal processing": "signal processing & computer systems",
    "silicon nanowires": "silicon nanowires",
    "single photon avalanche": "avalanche photodetectors",
    "social media": "data systems & social networks",
    "soft robotics": "soft robotics",
    "source imaging": "EEG source imaging",
    "speech recognition": "robust speech recognition",
    "spinal cord": "spinal cord stimulation",
    "stainless steel": "structural alloys",
    "storage systems": "storage systems",
    "structural health monitoring": "structural health monitoring",
    "technology advice congress": "science & technology policy",
    "testing object oriented software": "software engineering",
    "thermal conductivity": "thermal transport",
    "thermal expansion": "cryobiology & cryosurgery",
    "time reversal": "computational sensing & imaging",
    "transfer matrix": "design optimization & risk perception",
    "visual cortex": "visual cortex",
    "water networks": "infrastructure planning & optimization",
    "wave solutions": "nonlinear wave & flow dynamics",
    "wireless networks": "wireless networks",
    "wireless sensor networks": "wireless sensor networks",
}


def apply_reviewed_labels(
    extracted_by_cluster: Mapping[int, str],
    reviews: Mapping[str, str],
    *,
    reserved: set[str] | None = None,
) -> tuple[dict[int, str], dict[int, bool]]:
    """Apply exact editorial matches and reject ambiguous display labels."""

    labels = {
        cluster_id: reviews.get(extracted, extracted)
        for cluster_id, extracted in extracted_by_cluster.items()
    }
    used = set(reserved or ())
    for cluster_id in sorted(labels):
        label = labels[cluster_id]
        if not label or label in used:
            raise ValueError(
                "Topic display labels must be non-empty and unique; "
                f"cluster {cluster_id!r} resolved to {label!r}"
            )
        used.add(label)
    reviewed = {
        cluster_id: extracted in reviews
        for cluster_id, extracted in extracted_by_cluster.items()
    }
    return labels, reviewed


def review_catalog(level: int) -> Mapping[str, str]:
    """Return the reviewed phrase catalog for one hierarchy level."""

    catalogs: dict[int, Mapping[str, str]] = {
        0: REVIEWED_OVERVIEW_LABELS,
        1: REVIEWED_DETAIL_LABELS,
    }
    try:
        return catalogs[level]
    except KeyError as error:
        raise ValueError(f"Unknown topic hierarchy level: {level}") from error
