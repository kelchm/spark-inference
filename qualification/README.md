# Off-host qualification client

A stdlib-only Python client that prepares a deterministic qualification campaign offline, sends its requests to an explicit endpoint during an approved window, and judges the result offline from retained receipts and server logs. It was written for `glm53-mia-2026.11.07` on the `agent-1m` profile and is the harness whose receipts back a release's `qualification` entry.

It never starts, stops, restarts or kills a server, and never connects anywhere unless a command is given `--endpoint`. Home-lab performs every lifecycle operation by hand; the operator records each one with `record-lifecycle`.

| Command | Network | What it does |
| --- | --- | --- |
| `prepare` | none | Writes the campaign: `plan.json`, `manifest.json` (release, profile, image digest, learn directory, plan and harness hashes) and `prepared/` (the four PNGs and the two requests that need no fitting). |
| `server` | endpoint | Records `GET /health` and `GET /v1/models`. |
| `turn` | endpoint | Fits and measures the session's next request with `/v1/tokenize`, then sends it as a streamed chat completion. |
| `imprint` | endpoint | Posts the session's next continuation to `/v1/imprint/resume` (or `load`) and records where the server says it would resume. |
| `record-lifecycle` | none | Records a manual `start`, `restart`, `crash` or `eviction`, with identity artifacts such as `docker inspect` output of the running rank 0. |
| `import-log` | none | Retains a server log, by hash, for the verdict. |
| `verdict` | none | Judges the campaign. |

Run it from the repository root as `python3 -m qualification <command> …`; `--help` on each command lists its options. Tests: `python3 -m unittest tests.test_qualification` (loopback servers only).

## Engine contracts it relies on

Read from the engine source at `e52eb42b`, not guessed:

- `/v1/tokenize` takes a chat body (`messages`, `tools`, `reasoning_effort`, `add_generation_prompt`) and returns `count`, `max_model_len` and every token id, with images expanded as the chat route expands them. Token targets are only ever this count; nothing here estimates tokens from bytes or words.
- `/v1/imprint/resume` and `/v1/imprint/load` take the same body as the chat request and return `resume_from` (`ram`, `disk` or `none`), `ram_tokens`, `disk_tokens`, `disk_bytes` and, for `load`, `loaded_tokens`. The server answers 404 when it runs without `--learn-conversations`.
- The request body limit is 32 MiB. Prompts are limited to 1,048,576 − 32,768 = 1,015,808 tokens so a reply always fits the reservation.
- Disk-store log lines: a save is `imprint: N tokens on disk: …`, a read is `imprint: read N tokens from disk (…)`, a skipped or failed save is `imprint: saving N tokens skipped (…)`, `writing N tokens failed (…)` or `N tokens stay in RAM only (…)`, and the store opening is `glm: conversation states on disk in DIR (N states, N MiB of N MiB; N ranks' parts)`.

## Workload

All text comes from SHA-256 of the seed, so it is identical on every machine and the verdict can regenerate every request.

- **ledger**: filler prose in chunks of about 880 words, each with one planted fact (`the archive code for vault L0417 is KESTREL-4821-AQB`); codes are unique by construction. `first` is fitted to at least 300,000 measured tokens, `grow-1` to `grow-3` to 450k, 600k and 800k, and `grow-4` fills to the ceiling less the margin (default 24,576, which leaves room for three continuations) and must measure at least `near_floor` (default 980,000). Each turn asks about the first fact and two it just added.
- **evictor**: a separate ledger of at least 600,000 tokens, sent to push the ledger out of GPU memory before an eviction probe.
- **image / image-fork**: four 224×224 PNGs, each four quadrants of distinct named colors from a fixed palette. The fork keeps the text, dimensions and positions identical but rotates the colors. Run both before and after restart to expose image-identity aliasing. The verdict decodes the recorded pixels, so expected answers are mechanically known.
- **pi**: a Pi-shaped tool exchange. `call` must produce a `read_ledger` call; `result` replays that assistant message with its `reasoning_content` and the server's original call IDs exactly, plus the tool result, and must answer with the value only the tool supplied.
- Continuations (`turn --continue`) add one question to the ledger, image or pi conversation after a lifecycle event.

Answers are lines of `Qn=<answer>`; every one must match and none may repeat. Requests use `reasoning_effort` `low` and `max_tokens` 8,192 by default to bound replies; both are recorded in the plan and can be changed at `prepare`.

## Campaign

Prepare offline, before any window:

```sh
python3 -m qualification prepare --out <campaign> \
  --release recipes/glm53-mia/releases/glm53-mia-2026.11.07.json \
  --profile recipes/glm53-mia/sparkrun/profiles/glm53-mia-2026.11.07-agent-1m.yaml
```

In the approved window, with no other traffic on the server, and with `E=--campaign <campaign> --endpoint http://<rank0>:8888`:

1. Home-lab starts the server. `record-lifecycle --campaign <campaign> --event start --identity <rank0 docker inspect>`, then `server $E`.
2. Long context: `turn $E --session ledger` five times (`first` and `grow-1` … `grow-4`). Wait for the first ≥300K published-save receipt before sending the next turn; a later save cannot substitute for the first save.
3. `turn $E --session image`, `turn $E --session image-fork`, then `turn $E --session pi` twice.
4. Capture the rank 0 log with timestamps (`docker logs --timestamps …`) and `import-log` it. Home-lab restarts the server. `record-lifecycle --event restart --identity <new rank0 docker inspect>`.
5. For each of `ledger`, `image`, `image-fork` and `pi`: `imprint $E --session <s>`, then `turn $E --session <s> --continue`. Leave more than twice the clock-skew allowance between small probes (over four seconds at the default); ambiguous read lines prove neither probe.
6. Eviction: `turn $E --session evictor`, `record-lifecycle --event eviction`, `imprint $E --session ledger`, `turn $E --session ledger --continue`.
7. Crash: capture and import the log, home-lab kills rank 0 and brings the server back, `record-lifecycle --event crash --identity <rank0 docker inspect>`, then `imprint` and `turn --continue` for `ledger`.
8. Capture and import the final rank 0 log, then `verdict --campaign <campaign> --output <campaign>/verdict.json`.

Every probe writes its own receipt under `receipts/`, created exclusively and never overwritten; request and response wire bytes are kept beside it, gzipped and hashed. A failed request stops its session: run the step again with `--retry`, and the failure stays on record.

## What the verdict requires

The verdict re-parses every reply from its wire bytes, regenerates every request from the plan and the earlier replies, and reads token counts from the recorded `/tokenize` replies. Receipt fields such as `client_observed` are for the operator and are never trusted. A changed sidecar, artifact, plan or harness makes the campaign `invalid`.

- **Workload**, per step: HTTP 200, `[DONE]` after a `stop` (or, for `pi/call`, `tool_calls`) finish, positive usage, the exact planned request with every earlier reply replayed verbatim, the measured prompt within the step's target and the ceiling, and every answer right. A `length` finish, a partial or malformed stream, a deadline or a wrong answer fails the step; a later retry does not erase the failure.
- **Durability**, for each continuation after a recorded lifecycle event: a save line before the event covering at least the prior prompt less `--tolerance` (default 4,096 tokens) and no more than that prompt and its reply; a read line during the continuation covering at least that much and no more than the continuation's own prompt; for images, at least the measured image prefix, so the read includes the images. After a restart or crash, the store must reopen in the profile's learn directory with 2 ranks' parts and at least one state, and the identity artifact must name the release image digest. A skipped or failed save, a failed disk check, or a retained log that covers the continuation without a read fails the probe. A correct answer without a read is a cold prefill, not durability. A missing log, lifecycle record or identity artifact leaves the probe `unknown`. An eviction probe without a read is `unknown` unless the server said it would resume from disk.
- **Required probes**: ledger after restart, crash and eviction; image after restart; pi after restart.
- **Identity**: every `start`, `restart` and `crash` record names the release digest, and `/tokenize` reports `max_model_len` 1,048,576.

`qualification` summarizes only the automated workload, first-save, required resume probes and identity (`fail` over `unknown` over `pass`). Even a pass leaves `production=pending_manual_review`. It does not establish four-stream capacity, a verified mid-write crash, quota pressure, real Pi behavior or successful Python rollback. Review those separate home-lab receipts before following "Record qualification" in [AGENTS.md](../AGENTS.md). A lifecycle note alone is not proof of those operations.

If images are not reused from disk after a restart, the image probe fails and the release stays production-rejected, whatever else passed.

A separate `large_first_save` result requires a matching published save for the first measured ≥300K ledger prompt before the next chat. Missing evidence remains unknown; a skipped first save fails.

## Limits

- Log lines carry no request ID, so they are attributed by time window (±`--clock-skew`, default 2 s) and by token count. Run probes one at a time with nothing else using the server, keep both clocks on NTP, and capture logs with timestamps; untimestamped lines are counted but never attributed.
- The log of a container that is recreated is lost unless it is captured first: import the rank 0 log before every restart and after every crash.
- `cached_tokens` and other prompt usage counters are recorded but never used as evidence; the engine's accounting is known to be imperfect.
- The first save of a large prompt may be skipped (staging and RAM cache limits). The verdict records that as a failed probe for that conversation, not as a pass.
- A ledger turn's fit takes six to ten `/tokenize` calls, each sending the whole conversation (up to about 5 MiB) and receiving every token id (up to about 8 MiB). Default deadlines are 3,600 s for chat, 900 s for tokenize and imprint and 30 s for `server`; change them with `--deadline` and `--tokenize-deadline`.
- Tests use a loopback fake engine. The client has not yet run against the real engine.
