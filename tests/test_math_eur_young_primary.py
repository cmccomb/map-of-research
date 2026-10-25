"""Newly verified Math Scholar accounts retain their official visual home."""

from map_of_research.registry import load_registry


def test_eur_and_young_have_reviewed_primary_math_department() -> None:
    registry = load_registry()
    people = {person.person_id: person for person in registry.people}
    for person_id, profile_url in (
        (
            "person-christophereur-92d8e4ba",
            "https://www.cmu.edu/math/people/faculty/eur.html",
        ),
        (
            "person-michaelyoung-794f4658",
            "https://www.cmu.edu/math/people/faculty/young.html",
        ),
    ):
        person = people[person_id]
        assert person.primary_department_id == "math"
        assert person.primary_department_source_url == profile_url
        assert person.primary_department_verified_at == "2026-10-25"
        assert person.primary_department_basis == "reviewed"
        primary = registry.primary_affiliation(person_id)
        assert primary.home_type == "department"
        assert primary.home_id == "math"
        assert primary.school_id == "mcs"
