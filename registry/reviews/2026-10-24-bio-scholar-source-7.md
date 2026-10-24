# Biological Sciences Scholar source review, En Cai — 2026-10-24

This offline review covers all **20 visible rows** in En Cai's saved Google Scholar profile (`ySlRYXoAAAAJ`). The raw JSON and sanitized HTML are paired at `work/scholar-bootstrap/en-cai-visible-20-2026-10-24.{json,html}`. Both contain 20 publication rows; all 20 citation IDs are unique, the final **Show more** button is present but disabled, and the capture reports no page error or challenge. All captured rows have Scholar years. This establishes a complete *visible profile at capture time*, not a claim that Scholar includes every work or will remain unchanged.

| Raw artifact | SHA-256 |
| --- | --- |
| JSON | `d8a35d013a683c1c60a855f3de4e99e78a576848faa85652394681e1f35b6c2d` |
| Sanitized HTML | `c488e1fc2ec2e5ef2d9f2fda196b399b1e3dcadecf91295bfe0a052ef0041dcf` |

[CMU identifies Cai as an assistant professor in Biological Sciences](https://www.cmu.edu/bio/people/faculty/cai.html), supporting Bio as her primary department. The two exact `(scholar_id, author_pub_id)` decisions below are in [`data/curated/scholar-exclusions.json`](../../data/curated/scholar-exclusions.json). They affect map eligibility only. All 20 raw Scholar observations remain in the separate capture lane. This PR does not edit caches, rewrite Scholar fields, merge citation counts, or publish a snapshot.

## Exact exclusions

- `ySlRYXoAAAAJ:foquWX3nUaYC` appends a journal reference, `Science. 356: eaal3118`, to the complete title of the same 2017 article retained as `ySlRYXoAAAAJ:BwyfMAYsbu0C`. [CMU's Cai publication list](https://www.cmu.edu/bio/people/faculty/cai.html) identifies the single article with En Cai as first author; the [author-hosted original *Science* PDF](https://www.krummel.org/images/Publications/cai-pub-2017-01.pdf) confirms DOI `10.1126/science.aal3118`.
- `ySlRYXoAAAAJ:xtoqd-5pKcoC` has the title *Ya. Fiyala, J. Liu, PA Dowben*, the final three authors of the 2008 *Applied Physics Letters* article already retained as `ySlRYXoAAAAJ:UeHWp8X0CEIC`. [CMU's Cai publication list](https://www.cmu.edu/bio/people/faculty/cai.html) identifies one article at 92:122107 and names the complete author list; [coauthor Jing Liu's university publication list](https://manhattan.edu/directory/jing.liu) independently gives that same title, author sequence, and page.

## Retained observations and limits

The 2008 condensed-matter papers are genuine earlier work by this En Cai: [CMU lists both articles and their full author lists](https://www.cmu.edu/bio/people/faculty/cai.html), and [the University of Nebraska repository names En Cai on the CoS2 paper](https://digitalcommons.unl.edu/physicsdowben/242/). A change in research area is therefore no basis for excluding them. The 2026 niclosamide article is also hers: [the publisher gives `encai@andrew.cmu.edu` and Biological Sciences, CMU, as En Cai's affiliation](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2026.1761715/full).

The 2021 bioRxiv and 2022 *Journal of Cell Biology* chimeric-antigen-receptor observations, and the 2022 bioRxiv and 2022 *PNAS* membrane-scanning observations, remain as distinct preprint and journal forms. The 2026 bioRxiv T-cell membrane study and 2023 methods chapter also remain. No further false attribution or nonpublication decision met the exact source-evidence threshold among the 20 rows.
