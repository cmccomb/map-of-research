# McComb coauthor binding continuity at Hub 5073

The no-write release preview against immutable Hub parent
`5073c24450e2f239ca49f674cbc81287b915045b` and source snapshot SHA
`51ebf5c5b5185ba1ac4d19beb29fccebca9d89c8e4628d8f8c9df2db839028a3`
failed closed before any Hub write. Its region receipt reported exactly three
still-visible Christopher McComb Scholar observations whose work IDs change
when the reviewed Nikolas Martelaro and Burcu Akinci curated DOI observations
join those works. There were no unexplained visible-observation losses or gains.

| McComb Scholar observation suffix | Old work ID | New work ID | Curated DOI and coauthor |
| --- | --- | --- | --- |
| `2tRrZ1ZAMYUC` | `work-8de433ddb3e949b5b880ebc5f35e5ffb57c6858637a16880763b9689411ded16` | `work-6fa3b8e235d1e393aafcd58cb14d931cb72b403518d6b27b501c31f3d670492d` | `10.48550/arXiv.2609.18029`, Burcu Akinci |
| `ODE9OILHJdcC` | `work-2257e9ec8ef231c41955ffb5a044ee3d6a1688193d83e1fc85576af4e124ffb8` | `work-e422b73064730da7137e3aa7877cbd8585e1e069b169043323dc3003706fbc7b` | `10.1115/DETC2025-168883`, Nikolas Martelaro |
| `yMeIxYmEMEAC` | `work-f6c2ca1e38fc7affb2f0f78dd6f161b78d7080e348b8c08c0cbdf804b45bf8be` | `work-c26a95fc00552447ef52e0917741b667e12efeb96d9154dc831ed34423d34d8b` | `10.1017/dsj.2025.10008`, Nikolas Martelaro |

Both old and new observations remain map and collaboration eligible. The
approval is limited to this parent, the exact snapshot content, curated work
file, McComb cache, two source review ledgers, and the three listed binding
tuples. The publisher's region gate continues to reject any extra binding
change or unused approval. The source commit SHA can change when the source PR
is merged; content hashes remain the release authority.

Source evidence and authorship review:

- [Martelaro curated coauthorship](2026-10-30-mccomb-martelaro-curated-coauthorship.md)
- [Akinci curated coauthorship](2026-10-30-mccomb-akinci-curated-coauthorship.md)

No-write failure receipt:
`outputs/martelaro-burcu-final-no-write-preview-v2-2026-10-30/receipt.json`
in the task workspace. The replacement no-write preview and independent
staged-artifact audit are required before source merge or upload.
