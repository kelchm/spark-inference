# SparkRun qualification profile

`profiles/glm53-mia-2026.11.07-agent-1m.yaml` targets the published release by digest. It is a qualification profile, not an approved production deployment. Home-lab owns the operating window, host configuration, lifecycle and rollback. Nothing in this directory contacts a Spark unless an operator explicitly gives it to SparkRun during an approved window.

Generate and verify the profile, served template and mod inventory with:

```sh
python3 tools/render_sparkrun.py recipes/glm53-mia/releases/glm53-mia-2026.11.07.json
python3 tools/render_sparkrun.py recipes/glm53-mia/releases/glm53-mia-2026.11.07.json --check
python3 -B -m unittest discover -s tests -v
```

The runtime adapter in `plugins/tensorfold` is byte-identical to home-lab's currently maintained SparkRun 0.4.2 adapter. The actual SDK test resolves the external profile and its adjacent mod and translates both TP ranks while forbidding external process and connection attempts. It does not execute SparkRun's host lifecycle, Docker, a native binary or a model. The `.yaml` profile loads from an immutable materialized checkout; its mod resolves locally rather than falling back to a moving registry.

The mod installs a rank launcher and readiness/capacity scripts. It never patches engine source or replaces the released native binary. It uses Mia's exact template and the same guarded `EFFORT_LINE=thinking` rewrite as her launcher. This base advertises `max,high,low`; its discovery heuristic omits `none` because the template has no switch variable. The engine still closes a dangling think opener for explicit thinking-off requests. The readiness gate checks the actual advertised list and effort rendering, four concurrent answers and a tool call. Video support is absent from this image; unsupported video requests are not promised to return HTTP 400.

## Effective settings

| Setting | Selected value |
| --- | --- |
| Streams / total context / default reply budget | 4 / 1,048,576 / 32,768 |
| KV / paging / sharding / cache priority | FP8 / on / on / cache first |
| Requested device prompt cache / pinned host tier | 6 GiB per rank / off |
| Dense / head | q8 / q8, matching Mia's reported pair; her generic head default is BF16 |
| Thinking / effort / clearing / sampling | on / xhigh (template Max) / keep / top-k 20 |
| Memory reserve / prefill share | 12 GiB / 0.68 |
| Learned-store quota / minimum free / daily writes | 64 GiB / 50 GiB / 128 GiB |

The learned-store quota and daily write budget are separate limits. The larger explicit daily allowance avoids silently stopping this bounded campaign at the engine's 32 GiB/day default. The free-space floor retains the existing Python profile's 50 GiB protection; the approved-window preflight must verify enough free disk and record the actual filesystem. A dedicated release-specific store under `/cache/huggingface/spark-inference/learn/` never shares Python's `tensorfold-spill` directory.

The generated `glm53-mia-2026.11.07-quota-probe.yaml` changes only its profile/store name and quota to 1 GiB. Run it separately with no other inference engine present to exercise quota pressure without filling a 64 GiB store or host disk. Its result establishes the mechanism at the smaller quota, not a measured fill of the normal store. Both profiles pin the same image and use the same free-space and daily-write protections.

Both ranks' state parts are stored on **rank 0's NVMe** and routed over the TP link. KV sharding does not make this a per-Spark filesystem store. Record the worker's routed part and the head's actual quota, bytes and disk-read receipts; do not require fictitious worker files.

SparkRun admits rank 1 after the head's `MASTER_PORT` listens. GLM opens its watch line at NCCL port plus one before NCCL can join, so the launcher sets `NCCL_COMM_ID` to the detected fabric address and `MASTER_PORT-1`. Check both ports for collisions during the approved preflight. The local memory guard follows Mia's available-plus-parked-page rule, with a 2.5 GiB floor held for four seconds. It controls only this rank's process group. It is an emergency guard, not evidence of host-wide recovery.

## What startup proves

The capacity hook requires an unambiguous startup pool/cache receipt, at least the historical 2,582,528-token aggregate pool, and at least a 1M window plus verification room in each sharded part. `/health` must expose two KV parts with enough pages each. A configured context or aggregate pool alone is insufficient for a long sharded request.

The requested 6 GiB cache is a cap, not guaranteed funding. The hook's 2 GiB cache minimum is only a launch smoke threshold; the campaign must prove the actual >=300K first save and larger states. It also requires the learned store to open with the selected 64 GiB quota and both ranks' parts. Opening a store does not establish persistence. Correct durable images, large first saves, eviction/restart/process-crash recovery, bounded retention and real Pi behavior all remain qualification work.

The immutable upstream template origin is recorded in the generated mod manifest. Runtime adapter/readiness code was extracted from the user's home-lab tree; the rank launcher adapts its native candidate launcher. Their process/placement checks and Mia's launcher semantics are retained without adopting any of the retired engine patches.
