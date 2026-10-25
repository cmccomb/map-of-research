# Four exact Bio publication identities after the region repair

These reviews retain the Google Scholar observations as captured. The three
DOI aliases resolve exact Scholar citation IDs to published records, and the
metadata corrections replace only effective bibliographic fields after
checking the original Scholar title, year, venue, and authors. A fourth
DOI-free article has its own verified native source row in the staged source
snapshot. None of these reviews changes raw Scholar records.

| Scholar observation | Published identity and evidence | Effective display |
| --- | --- | --- |
| `FXqaMl4AAAAJ:UebtZRa9Y70C` | Neuron, DOI `10.1016/j.neuron.2017.06.006`; [publisher PII](https://www.sciencedirect.com/science/article/pii/S0896627317305044), [PubMed](https://pubmed.ncbi.nlm.nih.gov/28641104/), [DOI record](https://doi.org/10.1016/j.neuron.2017.06.006), and [CMU Barth bibliography](https://labs.bio.cmu.edu/barth/publications/) | “Constructing the External World”; Jiseok Lee and Alison L. Barth; *Neuron* 94(6), 1048–1050 (2017) |
| `28yZqbcAAAAJ:UeHWp8X0CEIC` | Molecular Cell, DOI `10.1016/j.molcel.2008.06.010`; [Cell version of record](https://www.cell.com/molecular-cell/fulltext/S1097-2765(08)00428-0), [PubMed](https://pubmed.ncbi.nlm.nih.gov/18614041/), and [DOI record](https://doi.org/10.1016/j.molcel.2008.06.010) | “Getting the message out”; C. Joel McManus and Brenton R. Graveley; *Molecular Cell* 31(1), 4–6 (2008) |
| `6ArMETUAAAAJ:nroGzMJTTpEC` | PLOS Computational Biology, DOI `10.1371/journal.pcbi.1014780`; [publisher version](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014780) | “Ten simple rules for suggesting reviewers during a paper submission”; Russell Schwartz, Scott Markel, Patricia M. Palagi, and B. F. Francis Ouellette; *PLOS Computational Biology* 22(9), e1014780 (2026) |
| `UZf8gAUAAAAJ:u5HHmVD_uO8C` | DOI-free *Journal of Undergraduate Neuroscience Education* article, `pmcid:pmc3970999`; [original journal PDF](https://www.funjournal.org/wp-content/uploads/2015/09/june-12-159.pdf), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3970999/), and [PubMed](https://pubmed.ncbi.nlm.nih.gov/24693264/) | “Controversies in Neuroscience: A Literature-Based Course for First Year Undergraduates that Improves Scientific Confidence While Teaching Concepts”; Amanda M. Willard and D. J. Brasier; 12(2), A159–A166 (2014) |

The McManus article has a 23-character title, and the Barth article has a
four-word title. Their short-title exceptions are restricted to the exact
Scholar observations, titles, years, DOIs, dated reviews, and publisher
PII/PubMed/DOI evidence chains. The strict guard is implemented separately in
the staged region-policy PR. On the frozen Chemistry weak-region layout,
each article is withheld without its own flag and retained with it. The
other 507 works in their shared region do not change eligibility.

The raw McManus observation `28yZqbcAAAAJ:HKviVsUxM5wC` has a poster-number
placeholder, no captured year, and no verified full byline. The meeting's
[poster list](https://rustbeltrna.org/2021/posters.php) suggests a real poster,
so this review neither labels it a nonpublication nor grants it a DOI or
metadata correction. It remains raw and unmapped pending exact attribution.

These additions follow the standalone Bernhard region repair. They require a
new full-corpus preview and all-profile release gate before publication.
