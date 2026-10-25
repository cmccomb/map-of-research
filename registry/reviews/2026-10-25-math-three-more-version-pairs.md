# Three more exact Math Scholar version pairs

Reviewed 2026-10-25. The six rows below are retained in the complete Google
Scholar profile captures. Each pair has the same authors and article title in
an arXiv record and a journal record. Their Scholar publication years differ,
so the default title/year matcher previously assigned two work identities.

| Researcher | Preprint Scholar row | Journal Scholar row | Published DOI |
| --- | --- | --- | --- |
| Prasad Tetali | FmZaK1wAAAAJ:zA6iFVUQeVQC (2014) | FmZaK1wAAAAJ:bFI3QPDXJZMC (2015) | 10.5802/afst.1464 |
| Sergey Nadtochiy | cAiedtsAAAAJ:7PzlFSSx8tAC (2022) | cAiedtsAAAAJ:dhFuZR0502QC (2023) | 10.1137/22M1490156 |
| Tom Bohman | SHxT0AEAAAAJ:pyW8ca7W8N0C (2022) | SHxT0AEAAAAJ:g5m5HwL7SMYC (2024) | 10.1017/fms.2024.6 |

The [Tetali arXiv record](https://arxiv.org/abs/1409.8605) names Matthias
Erbar, Jan Maas, and Prasad Tetali in 2014. The [journal's Numdam record](https://numdam.org/articles/10.5802/afst.1464/)
names the same authors and title in 2015, volume 24(4), pages 781–800, with
the published DOI.

The [Nadtochiy arXiv record](https://arxiv.org/abs/2204.05403) names Guillermo
Alonso Alvarez, Sergey Nadtochiy, and Kevin Webster in 2022. The
[SIAM article](https://epubs.siam.org/doi/10.1137/22M1490156) has the same
authors, title, and abstract, and appears in the 2023 journal volume 14(3),
pages 855–878. [SIAM-deposited Crossref metadata](https://api.crossref.org/works/10.1137/22M1490156)
independently exposes the exact authors, DOI, journal volume, issue, pages,
and 2023 publication year.

The [Bohman arXiv record](https://arxiv.org/abs/2208.00117) names Tom Bohman
and Jakob Hofstad in 2022. The [Cambridge journal article](https://doi.org/10.1017/fms.2024.6)
has the same authors and title in *Forum of Mathematics, Sigma* 12 (2024),
e24, with the published DOI.

Each preprint row has a reviewed preprint_of alias, and each journal row has a
reviewed version_of_record alias. The producer validates the exact captured
identity, title, year, and exclusion state before normalization. All six raw
Scholar observations, their source dates, and their citation IDs remain
unchanged. Each pair normalizes to one DOI work, with the journal row supplying
the work's publication year and venue.
