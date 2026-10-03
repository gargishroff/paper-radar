# Data sources: checked values

PLAN §4 asks for the rate limits and terms actually in force, with the date they were checked.

| Source | Checked | Value | How checked |
|---|---|---|---|
| arXiv OAI-PMH endpoint | 2026-09-27 | `https://oaipmh.arxiv.org/oai` (the old `export.arxiv.org/oai2` has been replaced) | `ListSets` request; arXiv's OAI help page |
| arXiv OAI-PMH sets | 2026-09-27 | `cs:cs:LG`, `cs:cs:CL`, `cs:cs:AI`, `cs:cs:CV`, `stat:stat:ML` all exist; records list every category they are in, so cross-lists are included | `ListSets` response |
| arXiv OAI-PMH pacing | 2026-09-27 | We wait 3 s between requests and honour `503 Retry-After`. A whole day for the 5 sets took ~13 s | Live run |
| arXivRaw format | 2026-09-27 | Carries the full version history with dates (needed to tell new papers from revisions); authors are a free-text string | Live `ListRecords` response |
| Cohere trial key | 2026-09-27 | ~1,000 calls/month across endpoints; ≤ 96 texts per embed call. Published per-minute embed limits disagree (2,000 inputs/min vs 5 calls/min) | Cohere rate-limit docs and third-party summaries |
| Cohere trial embed, measured | 2026-09-27 | 3 calls of 96 abstracts within ~30 s succeeded; the 4th got `429`. Each call bills ~27k tokens, so this fits a ~100k tokens/min cap (**inferred, not documented**). We now allow 3 calls/min; the 2026-09-23 run at that pace had no 429s | Three live daily-job runs (`api_usage`, job logs) |

## Measured volume

Each row is one `python -m radar.jobs.daily --no-embed` run (see `job_runs`). "New" = first version submitted ≤ 7 days before the datestamp.

| OAI datestamp | Records in the 5 sets | New submissions | Of which primary cs.LG / cs.CV / cs.CL / cs.AI |
|---|---|---|---|
| 2026-09-23 | 813 | 392 | not broken down |
| 2026-09-24 | 717 | 394 | 89 / 89 / 61 / 48 |
| 2026-09-25 | 827 | 466 | not broken down |
| 2026-09-26 (Sat) | 0 | 0 | arXiv does not announce on weekends |

No paper was counted as new on both days: the 2026-09-25 run inserted all 827 records as new rows.

At 96 texts per call, 394–466 new papers need 5 embed calls per day. Measured cost: 392 + 394 + 466 = 1,252 papers billed 352,794 input tokens (≈ 282 tokens per title+abstract).

## arXiv export API reliability

| Checked | Observation | Response |
|---|---|---|
| 2026-09-27/28 | `export.arxiv.org/api/query` returned `406` to the app for ~5 minutes at a time, while the same URL succeeded from curl, and recovered on its own. Neither the User-Agent, the comma encoding nor `Accept-Encoding` explained it. | Batches of ≤ 10 IDs use OAI-PMH `GetRecord`; larger batches try the export API 3 times, then fall back to OAI-PMH. |
| 2026-09-27 | Semantic Scholar's author endpoint sometimes returns `429` without an API key. | Exponential backoff; `SEMANTIC_SCHOLAR_API_KEY` is optional. |

## Embedding similarity baseline

Measured 2026-09-28 on Embed 4 (1024-d) vectors of 400 papers announced on 2026-09-25 (79,800 pairs).

| Statistic | Cosine |
|---|---|
| Random pair, median | 0.22 |
| Random pair, 90th percentile | 0.31 |
| Random pair, 99th percentile | 0.40 |
| Nearest neighbour within the day, median | 0.43 |

The 99th percentile (0.40) is the same-topic threshold for grouping profiles of fewer than 4 papers (`SAME_TOPIC_COSINE`).
