# McManus poster point: one-time historical loss approval

The complete 62-row Scholar profile keeps poster 101 as a raw observation. Its
four-name Scholar citation remains provisionally held from the public map and
coauthor graph because the organizer's public poster list does not verify the
complete byline or contributor roles. The existing public dataset mapped this
one observation, so an ordinary provisional hold must not silently remove its
point.

The reviewed manifest at `data/curated/bio-mcmanus-poster-loss-approval.json`
is bound to the post-Chemistry dataset parent
`dcbdfacd7e8cfeaefd4c91081c4b347b7c690061` and the complete Bio W3
source SHA-256 `d08fb3a422df8dee3b0d0f75a29d68fef9b7121229a9cf6d418917e1d4a19550`.
It uses schema version 1 and the exact keys returned by `approval_document()`
in `map_of_research/poster_loss_approval.py`. It records:

- the 40-character immutable Hub parent revision;
- the exact raw source snapshot SHA-256;
- SHA-256 of the reviewed profile-release and provisional-hold JSON files;
- the validated McManus hold, including its exact reason and Scholar citation
  URL.

The code fixes the only permitted person, Scholar profile, observation, prior
work, raw record hash, 62-row cache hash, held ID hash, and capture row index.
The manifest names exactly one expected loss. The publisher independently
checks the parent row was visible, the candidate row is held, and the map gate
has no other unapproved losses. The Bio prewrite gate also requires the
approved-loss receipt and checks the poster contributes no public point or
authorship.

The upload and no-write preview pass the manifest only while it exists. After
the first successful upload, remove this file before any later dataset release.
A stale parent or a second run with no actual poster loss fails before any Hub
write.
