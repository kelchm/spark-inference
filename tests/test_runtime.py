"""Offline unit tests for the shared container runtime/ contract.

Covers:
- 3-way mount contract (HF model cache, runtime inputs/drafter/chat template, persistent learn store)
- Parameterized RDMA HCA & fabric env vars (SparkRun + Kubernetes LeaderWorkerSet, zero hardcoded host interfaces)
- MemGuard unified-memory accounting (/proc/meminfo + /proc/zoneinfo parked pages + cgroup v2 caps + sustained breach)
- PID-1 entrypoint signal forwarding (Rank 0 drain, Rank 1 orderly SIGTERM flush & second-signal / timeout escalation,
  SIGHUP forwarding, zombie reaping, MemGuard exit 72/73)
- Health / readiness / worker probe wrapper (open health, keyed minimal health, 503 draining, capacity 2,582,528 floor,
  worker state probe, secret redaction)
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import contract, entrypoint, health, memguard  # noqa: E402


def _base_env(tmp: Path, *, rank: str = "0") -> dict[str, str]:
    hf = tmp / "hf"
    inputs = tmp / "inputs"
    learn_root = tmp / "learn"
    learn_dir = learn_root / "glm53-kelchm-test"
    model = hf / "hub" / "models--glm53"
    drafter = inputs / "dflash2"
    model.mkdir(parents=True, exist_ok=True)
    drafter.mkdir(parents=True, exist_ok=True)
    learn_root.mkdir(parents=True, exist_ok=True)
    return {
        "RANK": rank,
        "WORLD_SIZE": "2",
        "FABRIC_MASTER": "198.19.240.11",
        "MASTER_PORT": "25000",
        "NCCL_IB_HCA": "=rocep1s0f0:1,roceP2p1s0f0:1",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
        "HF_HOME": str(hf),
        "TF_RUNTIME_INPUTS_DIR": str(inputs),
        "MODEL_PATH": str(model),
        "DRAFTER": str(drafter),
        "GLM_LEARN_ROOT": str(learn_root),
        "GLM_LEARN_DIR": str(learn_dir),
        "GLM_LEARN_GIB": "64",
        "GLM_LEARN_MIN_FREE_GIB": "50",
        "GLM_LEARN_DAILY_GIB": "128",
        "TF_KV_PAGES": "on",
    }


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sysfs = self.tmp / "sysfs_ib"
        self.sysfs.mkdir()
        (self.sysfs / "rocep1s0f0").mkdir()
        (self.sysfs / "roceP2p1s0f0").mkdir()
        (self.sysfs / "mlx5_0").mkdir()
        (self.sysfs / "mlx5_1").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_sparkrun_tp2_launch_spec_and_three_way_mounts(self):
        env = _base_env(self.tmp, rank="0")
        tmpl = Path(env["TF_RUNTIME_INPUTS_DIR"]) / "chat_template.jinja"
        tmpl.write_text("{{ messages }}")
        spec = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)

        self.assertEqual(spec.rank, 0)
        self.assertEqual(spec.world, 2)
        self.assertEqual(spec.master_ipv4, "198.19.240.11")
        self.assertEqual(spec.nccl_port, 24999)
        self.assertEqual(spec.rdma.hcas, ("rocep1s0f0", "roceP2p1s0f0"))
        self.assertEqual(spec.rdma.gid_index, 3)
        self.assertTrue(spec.rdma.enabled)
        self.assertEqual(spec.env["NCCL_COMM_ID"], "198.19.240.11:24999")
        self.assertEqual(spec.env["TF_ROCE_HCA"], "rocep1s0f0,roceP2p1s0f0")
        self.assertEqual(spec.env["TF_ROCE_GID_INDEX"], "3")
        self.assertEqual(spec.env["TF_GLM_CLEAR_THINKING"], "0")
        self.assertEqual(spec.env["TF_KV_PAGES"], "on")
        self.assertNotIn("TF_GLM_RANK", spec.env)
        self.assertNotIn("TF_GLM_WORLD", spec.env)
        self.assertIn("--tp", spec.argv)
        self.assertIn("--master", spec.argv)
        self.assertIn("24999", spec.argv)
        self.assertIn("--chat-template", spec.argv)
        self.assertIn(str(tmpl), spec.argv)
        self.assertIn("--learn-conversations", spec.argv)
        self.assertTrue(Path(env["GLM_LEARN_DIR"]).is_dir())

    def test_kubernetes_leaderworkerset_env_runs_same_contract_unchanged(self):
        hf = self.tmp / "k8s-hf"
        inputs = self.tmp / "k8s-inputs"
        pvc = self.tmp / "pvc-learn"
        model = hf / "glm53"
        drafter = inputs / "dflash2"
        model.mkdir(parents=True)
        drafter.mkdir(parents=True)
        pvc.mkdir(parents=True)
        k8s_env = {
            "LWS_WORKER_INDEX": "1",
            "LWS_GROUP_SIZE": "2",
            "LWS_LEADER_ADDRESS": "glm53-0.glm53.inference.svc.cluster.local",
            "MASTER_PORT": "29551",
            "TF_TP_PORT_OFFSET": "0",
            "TF_ROCE_HCA": "mlx5_0,mlx5_1",
            "TF_ROCE_GID_INDEX": "2",
            "HF_HOME": str(hf),
            "TF_RUNTIME_INPUTS_DIR": str(inputs),
            "MODEL_PATH": str(model),
            "DRAFTER": str(drafter),
            "GLM_LEARN_ROOT": str(pvc),
            "GLM_LEARN_DIR": str(pvc / "glm53-prod"),
        }
        spec = contract.build_launch_spec(
            k8s_env,
            sysfs_ib_dir=self.sysfs,
            resolve_host=lambda host: "10.244.1.10" if host.endswith(".cluster.local") else "",
        )
        self.assertEqual(spec.rank, 1)
        self.assertEqual(spec.world, 2)
        self.assertEqual(spec.master_ipv4, "10.244.1.10")
        self.assertEqual(spec.nccl_port, 29551)
        self.assertEqual(spec.rdma.hcas, ("mlx5_0", "mlx5_1"))
        self.assertEqual(spec.rdma.gid_index, 2)
        self.assertEqual(spec.env["NCCL_COMM_ID"], "10.244.1.10:29551")
        # Rank 1 does not require learn PVC creation on worker node.
        self.assertFalse((pvc / "glm53-prod").exists())

    def test_rdma_hca_validation_and_exclusions(self):
        self.assertEqual(contract.parse_rdma_hcas("=mlx5_0:1,mlx5_1:1", sysfs_ib_dir=self.sysfs), ("mlx5_0", "mlx5_1"))
        for bad in ("^mlx5_0", "mlx5_0:2", "mlx5_0,mlx5_0", "mlx5_0,mlx5_1,rocep1s0f0", "missing_hca", "bad hca"):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse_rdma_hcas(bad, sysfs_ib_dir=self.sysfs)

    def test_mount_and_conflict_guardrails_fail_closed(self):
        base = _base_env(self.tmp)
        cases = [
            {"GLM_LEARN_DIR": base["GLM_LEARN_DIR"] + "/../escape"},
            {"GLM_LEARN_DIR": base["HF_HOME"]},
            {"GLM_LEARN_DIR": "/cache/huggingface/tensorfold-spill/run"},
            {"GLM_LEARN_DIR": "/tmp/outside-root"},
            {"GLM_LEARN_GIB": "0"},
            {"TF_KV_SHARE": "16"},
            {"GLM_LEARN_ENABLED": "0", "TF_KV_SHARE": "16", "TF_GLM_REQUEST_STARTS": "1"},
            {"TF_KV_SHARE": "bogus"},
            {"VISION_VIDEO_TOKENS": "16"},
            {"EXTRA_ARGS": "--context 128"},
            {"NCCL_IB_HCA": "", "TF_ROCE_HCA": ""},
            {"TF_KV_PAGES": "bogus"},
            {"MASTER_PORT": "1"},
        ]
        for override in cases:
            with self.subTest(override=override):
                env = dict(base, **override)
                with self.assertRaises(ValueError):
                    contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)

    def test_legacy_launch_style_and_no_roce_mode(self):
        env = _base_env(self.tmp, rank="0")
        env.update(TF_TP_LAUNCH_STYLE="legacy", TF_ROCE="0", NCCL_IB_HCA="")
        spec = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)
        self.assertFalse(spec.rdma.enabled)
        self.assertEqual(spec.env["TF_ROCE"], "0")
        self.assertEqual(spec.env["TF_GLM_RANK"], "0")
        self.assertEqual(spec.env["TF_GLM_WORLD"], "2")
        self.assertNotIn("--tp", spec.argv)

    def test_secret_redaction_and_argv_sanitization(self):
        raw = "Authorization: Bearer sk-super-secret-9999 failed with token sk-super-secret-9999"
        redacted = contract.redact_secrets(raw, env={"TENSORFOLD_API_KEY": "sk-super-secret-9999"})
        self.assertNotIn("sk-super-secret-9999", redacted)
        self.assertIn("<redacted>", redacted)

        sanitized = contract.sanitize_argv(
            ["tensorfold-native", "serve", "--api-key", "sk-cli-secret", "--api-key=sk-inline-secret"],
            env={"TENSORFOLD_API_KEY": "sk-env-secret"},
        )
        self.assertNotIn("sk-cli-secret", " ".join(sanitized))
        self.assertNotIn("sk-inline-secret", " ".join(sanitized))

    def test_contract_refuses_comm_id_mismatch_port_collision_symlink_escape_and_protected_extra_args(self):
        base = _base_env(self.tmp, rank="0")
        # Conflicting NCCL_COMM_ID
        with self.assertRaisesRegex(ValueError, "NCCL_COMM_ID"):
            contract.build_launch_spec(dict(base, NCCL_COMM_ID="10.0.0.9:24999"), sysfs_ib_dir=self.sysfs)
        # Matching NCCL_COMM_ID is accepted
        spec_ok = contract.build_launch_spec(dict(base, NCCL_COMM_ID="198.19.240.11:24999"), sysfs_ib_dir=self.sysfs)
        self.assertEqual(spec_ok.env["NCCL_COMM_ID"], "198.19.240.11:24999")
        # HTTP port collides with NCCL bootstrap (24999) or tp_watch (25000)
        for bad_port in ("24999", "25000"):
            with self.assertRaisesRegex(ValueError, "collides"):
                contract.build_launch_spec(dict(base, PORT=bad_port), sysfs_ib_dir=self.sysfs)
        # Protected flags (including --prompt-cache-disk) refused even when ALLOW_EXTRA_ARGS=1
        for bad_extra in ("--rank 1", "--prompt-cache-disk /tmp/escape"):
            with self.assertRaisesRegex(ValueError, "contract-managed"):
                contract.build_launch_spec(
                    dict(base, ALLOW_EXTRA_ARGS="1", EXTRA_ARGS=bad_extra), sysfs_ib_dir=self.sysfs
                )
        # Symlink GLM_LEARN_DIR escaping GLM_LEARN_ROOT is refused
        outside = self.tmp / "outside-target"
        outside.mkdir()
        link_dir = Path(base["GLM_LEARN_ROOT"]) / "symlinked-store"
        link_dir.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            contract.build_launch_spec(dict(base, GLM_LEARN_DIR=str(link_dir)), sysfs_ib_dir=self.sysfs)

    def test_p3_knobs_and_vision_video_tokens_pass_through_with_learn_conversations(self):
        env = _base_env(self.tmp, rank="0")
        env.update(
            GLM_LEARN_ENABLED="1",
            TF_KV_SHARE="0",
            TF_GLM_REQUEST_STARTS="1",
            TF_GLM_MULTI_PREFILL="1",
            TF_GLM_ROCE_FUSE="1",
            TF_GLM_PRICED_WINDOWS="1",
            VISION_VIDEO_TOKENS="64",
        )
        spec = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)
        for knob in ("TF_GLM_REQUEST_STARTS", "TF_GLM_MULTI_PREFILL", "TF_GLM_ROCE_FUSE", "TF_GLM_PRICED_WINDOWS"):
            self.assertEqual(spec.env[knob], "1")
        self.assertIn("--learn-conversations", spec.argv)
        idx = spec.argv.index("--vision-video-tokens")
        self.assertEqual(spec.argv[idx + 1], "64")

        # Also verify explicit --vision-video-tokens via EXTRA_ARGS is not blocked.
        env_extra = _base_env(self.tmp, rank="0")
        env_extra.update(ALLOW_EXTRA_ARGS="1", EXTRA_ARGS="--vision-video-tokens 64")
        spec_extra = contract.build_launch_spec(env_extra, sysfs_ib_dir=self.sysfs)
        idx2 = spec_extra.argv.index("--vision-video-tokens")
        self.assertEqual(spec_extra.argv[idx2 + 1], "64")

    def test_explicit_off_flags_comma_separated_api_keys_and_symlinked_learn_ancestor(self):
        # 1. THINKING_ENABLED=0 emits --no-thinking; PROMPT_CACHE_GIB=0 still emits --prompt-cache-gib 0
        env = _base_env(self.tmp, rank="0")
        env.update(THINKING_ENABLED="0", PROMPT_CACHE_GIB="0")
        spec = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)
        self.assertIn("--no-thinking", spec.argv)
        self.assertNotIn("--thinking", spec.argv)
        self.assertNotIn("--reasoning-effort", spec.argv)
        pc_idx = spec.argv.index("--prompt-cache-gib")
        self.assertEqual(spec.argv[pc_idx + 1], "0")

        # 2. Comma-separated TENSORFOLD_API_KEY parses like auth.zig and redacts every member
        raw_keys = " , alpha-secret-1111 \t, \n beta-secret-2222 , "
        self.assertEqual(contract.parse_api_keys(raw_keys), ("alpha-secret-1111", "beta-secret-2222"))
        self.assertEqual(contract.primary_api_key(raw_keys), "alpha-secret-1111")
        scrubbed = contract.redact_secrets(
            "leaked alpha-secret-1111 and beta-secret-2222",
            env={"TENSORFOLD_API_KEY": raw_keys},
        )
        self.assertNotIn("alpha-secret-1111", scrubbed)
        self.assertNotIn("beta-secret-2222", scrubbed)

        # 3. Symlinked ancestor pointing at HF_HOME or tensorfold-spill is refused even when GLM_LEARN_ROOT is unset
        base_no_root = _base_env(self.tmp, rank="0")
        base_no_root["GLM_LEARN_ROOT"] = ""
        hf_alias = self.tmp / "alias_parent"
        hf_alias.symlink_to(Path(base_no_root["HF_HOME"]).parent)
        hf_name = Path(base_no_root["HF_HOME"]).name
        with self.assertRaisesRegex(ValueError, "forbidden"):
            contract.build_launch_spec(
                dict(base_no_root, GLM_LEARN_DIR=str(hf_alias / hf_name)), sysfs_ib_dir=self.sysfs
            )
        spill_real = self.tmp / "tensorfold-spill"
        spill_real.mkdir()
        spill_alias = self.tmp / "innocent_mount"
        spill_alias.symlink_to(spill_real)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            contract.build_launch_spec(
                dict(base_no_root, GLM_LEARN_DIR=str(spill_alias / "new-run")), sysfs_ib_dir=self.sysfs
            )
        self.assertFalse((spill_real / "new-run").exists())


class MemGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_parked_pages_and_memtotal_cap(self):
        meminfo = self.tmp / "meminfo"
        zoneinfo = self.tmp / "zoneinfo"
        # 10 GiB total, 2 GiB MemAvailable
        meminfo.write_text(f"MemTotal: {10 * 1024 * 1024} kB\nMemAvailable: {2 * 1024 * 1024} kB\n")
        # 4096-byte pages: count=1_048_576 (4 GiB), high_min=262_144 (1 GiB) => 3 GiB parked => 5 GiB effective
        zoneinfo.write_text(
            "Node 0, zone Normal\n"
            "  pagesets\n"
            "    cpu: 0\n"
            "              count: 1048576\n"
            "              high_min: 262144\n"
            "  all_unreclaimable: 0\n"
        )
        sample = memguard.sample_memory(
            meminfo,
            zoneinfo,
            page_size=4096,
            cgroup_max_path=self.tmp / "no_max",
            cgroup_current_path=self.tmp / "no_cur",
        )
        self.assertAlmostEqual(sample.mem_available_gib, 2.0, places=3)
        self.assertAlmostEqual(sample.parked_gib, 3.0, places=3)
        self.assertAlmostEqual(sample.effective_gib, 5.0, places=3)

        # Cgroup limit tighter than host effective memory wins.
        cmax = self.tmp / "memory.max"
        ccur = self.tmp / "memory.current"
        cmax.write_text(str(8 * memguard.GIB))
        ccur.write_text(str(7 * memguard.GIB))
        cg_sample = memguard.sample_memory(
            meminfo, zoneinfo, page_size=4096, cgroup_max_path=cmax, cgroup_current_path=ccur,
            proc_cgroup_path=self.tmp / "no_proc_cg", cgroup_root_path=self.tmp / "no_cg_root",
        )
        self.assertAlmostEqual(cg_sample.cgroup_available_gib, 1.0, places=3)
        self.assertAlmostEqual(cg_sample.effective_gib, 1.0, places=3)

        # Nested cgroup v2 hierarchy: unlimited leaf, tighter finite parent cap must be detected.
        cg_root = self.tmp / "cg_root"
        parent = cg_root / "kubepods.slice" / "pod1"
        leaf = parent / "ctr1"
        leaf.mkdir(parents=True)
        (leaf / "memory.max").write_text("max\n")
        (leaf / "memory.current").write_text(str(2 * memguard.GIB))
        (parent / "memory.max").write_text(str(6 * memguard.GIB))
        (parent / "memory.current").write_text(str(4500 * 1024 ** 2))  # ~1.6 GiB headroom
        proc_cg = self.tmp / "proc_cgroup"
        proc_cg.write_text("0::/kubepods.slice/pod1/ctr1\n")
        nested = memguard.sample_memory(
            meminfo, zoneinfo, page_size=4096,
            cgroup_max_path=self.tmp / "no_max", cgroup_current_path=self.tmp / "no_cur",
            proc_cgroup_path=proc_cg, cgroup_root_path=cg_root,
        )
        self.assertIsNotNone(nested.cgroup_available_gib)
        self.assertLess(nested.effective_gib, 2.0)

    def test_finite_cgroup_max_with_unreadable_or_malformed_current_fails_closed(self):
        meminfo = self.tmp / "meminfo"
        zoneinfo = self.tmp / "zoneinfo"
        meminfo.write_text(f"MemTotal: {128 * 1024 * 1024} kB\nMemAvailable: {100 * 1024 * 1024} kB\n")
        zoneinfo.write_text("")
        # 1. Direct path: finite memory.max with missing memory.current raises RuntimeError
        cmax = self.tmp / "finite.max"
        cmax.write_text(str(8 * memguard.GIB))
        with self.assertRaises(RuntimeError):
            memguard.sample_memory(
                meminfo, zoneinfo, page_size=4096,
                cgroup_max_path=cmax, cgroup_current_path=self.tmp / "missing.current",
                proc_cgroup_path=self.tmp / "no_proc_cg", cgroup_root_path=self.tmp / "no_cg_root",
            )
        # 2. Direct path: finite memory.max with malformed memory.current raises ValueError
        bad_cur = self.tmp / "bad.current"
        bad_cur.write_text("not-a-number\n")
        with self.assertRaises(ValueError):
            memguard.sample_memory(
                meminfo, zoneinfo, page_size=4096,
                cgroup_max_path=cmax, cgroup_current_path=bad_cur,
                proc_cgroup_path=self.tmp / "no_proc_cg", cgroup_root_path=self.tmp / "no_cg_root",
            )
        # 3. Hierarchy walk: finite ancestor memory.max with unreadable/malformed memory.current triggers sample_error
        cg_root = self.tmp / "cg_fail"
        parent = cg_root / "kubepods.slice" / "pod2"
        leaf = parent / "ctr2"
        leaf.mkdir(parents=True)
        (leaf / "memory.max").write_text("max\n")
        (parent / "memory.max").write_text(str(4 * memguard.GIB))
        # No parent/memory.current file exists
        proc_cg = self.tmp / "proc_cg_fail"
        proc_cg.write_text("0::/kubepods.slice/pod2/ctr2\n")
        guard = memguard.MemGuard(
            floor_gib=2.5,
            sustain_s=4.0,
            sampler=lambda: memguard.sample_memory(
                meminfo, zoneinfo, page_size=4096,
                cgroup_max_path=self.tmp / "no_max", cgroup_current_path=self.tmp / "no_cur",
                proc_cgroup_path=proc_cg, cgroup_root_path=cg_root,
            ),
        )
        status, reason = guard.check(0.0)
        self.assertEqual(status, "sample_error")
        self.assertIn("memory.current", reason or "")

    def test_sustained_breach_transient_recovery_and_sample_error(self):
        readings = iter([
            memguard.MemSample(2.0, 0.0, 2.0, None, 2.0),  # t=0: dip starts
            memguard.MemSample(2.1, 0.0, 2.1, None, 2.1),  # t=2: still below floor, <4s
            memguard.MemSample(4.0, 0.0, 4.0, None, 4.0),  # t=3: recovers, resets timer
            memguard.MemSample(1.8, 0.0, 1.8, None, 1.8),  # t=5: new dip starts
            memguard.MemSample(1.7, 0.0, 1.7, None, 1.7),  # t=9: sustained 4s -> low_memory
        ])
        guard = memguard.MemGuard(floor_gib=2.5, sustain_s=4.0, sampler=lambda: next(readings))
        self.assertEqual(guard.check(0.0)[0], "ok")
        self.assertEqual(guard.check(2.0)[0], "ok")
        self.assertEqual(guard.check(3.0)[0], "ok")
        self.assertEqual(guard.check(5.0)[0], "ok")
        status, reason = guard.check(9.0)
        self.assertEqual(status, "low_memory")
        self.assertIn("2.50 GiB", reason)

        broken = memguard.MemGuard(sampler=lambda: (_ for _ in ()).throw(OSError("proc gone")))
        self.assertEqual(broken.check(0.0)[0], "sample_error")


class EntrypointSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sysfs = self.tmp / "ib"
        self.sysfs.mkdir()
        (self.sysfs / "mlx5_0").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _spec_for_python_script(self, script: str, *, rank: str) -> contract.LaunchSpec:
        env = _base_env(self.tmp, rank=rank)
        env["NCCL_IB_HCA"] = "mlx5_0"
        base = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)
        return contract.LaunchSpec(
            rank=base.rank,
            world=base.world,
            master_raw=base.master_raw,
            master_ipv4=base.master_ipv4,
            master_port=base.master_port,
            nccl_port=base.nccl_port,
            launch_style=base.launch_style,
            mounts=base.mounts,
            rdma=base.rdma,
            argv=(sys.executable, "-c", script),
            env=base.env,
        )

    def test_rank0_forwards_sigterm_immediately_for_clean_drain(self):
        marker = self.tmp / "rank0_sig.txt"
        script = (
            "import signal, sys, time\n"
            f"p = {str(marker)!r}\n"
            "def on_term(s, f):\n"
            "    open(p, 'w').write(str(s))\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "while True: time.sleep(0.05)\n"
        )
        spec = self._spec_for_python_script(script, rank="0")
        signals: list[int] = []
        cfg = entrypoint.SupervisorConfig(
            shutdown_grace_s=5.0,
            memguard_enabled=False,
            memguard_interval_s=0.05,
            state_file=self.tmp / "state0.json",
        )

        def send_later():
            time.sleep(0.15)
            signals.append(signal.SIGTERM)

        t = threading.Thread(target=send_later, daemon=True)
        t.start()
        rc = entrypoint.supervise(spec, cfg, install_signals=False, external_signals=signals, log=lambda _: None)
        t.join(timeout=2)
        self.assertEqual(rc, 0)
        self.assertEqual(marker.read_text(), str(signal.SIGTERM))

    def test_rank0_forwards_second_sigterm_even_when_already_queued_before_stop(self):
        script = (
            "import signal, sys, time\n"
            "seen = []\n"
            "def on_term(s, f):\n"
            "    seen.append(s)\n"
            "    if len(seen) >= 2: sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "while True: time.sleep(0.02)\n"
        )
        spec = self._spec_for_python_script(script, rank="0")
        signals: list[int] = [signal.SIGTERM, signal.SIGTERM]
        cfg = entrypoint.SupervisorConfig(
            shutdown_grace_s=5.0,
            memguard_enabled=False,
            memguard_interval_s=0.05,
            state_file=self.tmp / "state0_queued2.json",
        )
        t0 = time.monotonic()
        # Wait briefly in child so handler is installed before signals fire: use a tiny script delay via thread
        signals.clear()

        def queue_both():
            time.sleep(0.15)
            # Enqueue both signals atomically before the next supervise poll tick
            signals.extend([signal.SIGTERM, signal.SIGTERM])

        t = threading.Thread(target=queue_both, daemon=True)
        t.start()
        rc = entrypoint.supervise(spec, cfg, install_signals=False, external_signals=signals, log=lambda _: None)
        t.join(timeout=2)
        self.assertEqual(rc, 0)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_rank1_holds_sigterm_during_flush_and_exits_cleanly_when_rank0_finishes(self):
        """Rank 1 must NOT receive SIGTERM while rank 0 is flushing disk cache; it exits cleanly on its own."""
        got_term = self.tmp / "rank1_got_term"
        flush_done = self.tmp / "rank1_flush_done"
        script = (
            "import pathlib, signal, sys, time\n"
            f"got = pathlib.Path({str(got_term)!r})\n"
            f"done = pathlib.Path({str(flush_done)!r})\n"
            "signal.signal(signal.SIGTERM, lambda s, f: (got.write_text('TERM'), sys.exit(143)))\n"
            "deadline = time.monotonic() + 5.0\n"
            "while not done.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.03)\n"
            "sys.exit(0)\n"
        )
        spec = self._spec_for_python_script(script, rank="1")
        signals: list[int] = []
        synced = []
        state_file = self.tmp / "state1.json"
        cfg = entrypoint.SupervisorConfig(
            rank1_flush_grace_s=3.0,
            rank1_kill_grace_s=1.0,
            memguard_enabled=False,
            memguard_interval_s=0.05,
            state_file=state_file,
        )

        phases_seen = []

        def orchestrate():
            time.sleep(0.15)
            signals.append(signal.SIGTERM)  # Orchestrator stops rank 1 concurrently with rank 0
            time.sleep(0.20)
            if state_file.is_file():
                phases_seen.append(json.loads(state_file.read_text()).get("phase"))
            # Simulate rank 0 completing disk flush and telling follower to leave cleanly
            flush_done.write_text("ok")

        t = threading.Thread(target=orchestrate, daemon=True)
        t.start()
        rc = entrypoint.supervise(
            spec,
            cfg,
            sync_fn=lambda: synced.append(True),
            install_signals=False,
            external_signals=signals,
            log=lambda _: None,
        )
        t.join(timeout=2)
        self.assertEqual(rc, 0)
        self.assertFalse(got_term.exists(), "Rank 1 engine was killed prematurely during flush window")
        self.assertEqual(phases_seen, ["flushing"])
        self.assertEqual(len(synced), 1)

    def test_rank1_escalates_when_flush_grace_expires(self):
        script = "import time\nwhile True: time.sleep(0.05)\n"
        spec = self._spec_for_python_script(script, rank="1")
        signals: list[int] = []
        cfg = entrypoint.SupervisorConfig(
            rank1_flush_grace_s=0.25,
            rank1_kill_grace_s=0.5,
            memguard_enabled=False,
            memguard_interval_s=0.05,
            state_file=self.tmp / "state1_timeout.json",
        )

        def send_later():
            time.sleep(0.1)
            signals.append(signal.SIGTERM)

        t = threading.Thread(target=send_later, daemon=True)
        t.start()
        rc = entrypoint.supervise(
            spec,
            cfg,
            sync_fn=lambda: None,
            install_signals=False,
            external_signals=signals,
            log=lambda _: None,
        )
        t.join(timeout=2)
        self.assertEqual(rc, 128 + signal.SIGTERM)

    def test_memguard_stops_rank_with_exit_73(self):
        script = "import time\nwhile True: time.sleep(0.05)\n"
        spec = self._spec_for_python_script(script, rank="0")
        low = memguard.MemSample(1.0, 0.0, 1.0, None, 1.0)
        guard = memguard.MemGuard(floor_gib=2.5, sustain_s=0.1, sampler=lambda: low)
        cfg = entrypoint.SupervisorConfig(
            memguard_enabled=True,
            memguard_interval_s=0.05,
            memguard_stop_grace_s=1.0,
            state_file=self.tmp / "state_oom.json",
        )
        rc = entrypoint.supervise(spec, cfg, mem_guard=guard, install_signals=False, log=lambda _: None)
        self.assertEqual(rc, 73)

    def test_memguard_sample_error_stops_rank_with_exit_72(self):
        script = "import time\nwhile True: time.sleep(0.05)\n"
        spec = self._spec_for_python_script(script, rank="0")
        guard = memguard.MemGuard(
            floor_gib=2.5,
            sustain_s=0.1,
            sampler=lambda: (_ for _ in ()).throw(OSError("meminfo unreadable")),
        )
        cfg = entrypoint.SupervisorConfig(
            memguard_enabled=True,
            memguard_interval_s=0.05,
            memguard_stop_grace_s=1.0,
            state_file=self.tmp / "state_err.json",
        )
        rc = entrypoint.supervise(spec, cfg, mem_guard=guard, install_signals=False, log=lambda _: None)
        self.assertEqual(rc, 72)

    def test_rank1_second_sigterm_cuts_flush_wait_short(self):
        script = "import time\nwhile True: time.sleep(0.05)\n"
        spec = self._spec_for_python_script(script, rank="1")
        signals: list[int] = []
        cfg = entrypoint.SupervisorConfig(
            rank1_flush_grace_s=10.0,  # would take 10s if not cut short
            rank1_kill_grace_s=0.5,
            memguard_enabled=False,
            memguard_interval_s=0.05,
            state_file=self.tmp / "state1_second.json",
        )

        def double_signal():
            time.sleep(0.1)
            signals.append(signal.SIGTERM)
            time.sleep(0.1)
            signals.append(signal.SIGTERM)

        started = time.monotonic()
        t = threading.Thread(target=double_signal, daemon=True)
        t.start()
        rc = entrypoint.supervise(
            spec, cfg, sync_fn=lambda: None, install_signals=False, external_signals=signals, log=lambda _: None
        )
        t.join(timeout=2)
        self.assertEqual(rc, 128 + signal.SIGTERM)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_entrypoint_dry_run_cli_and_disabled_learn_mode(self):
        env = _base_env(self.tmp, rank="0")
        env.update(GLM_LEARN_ENABLED="0", GLM_LEARN_DIR="", NCCL_IB_HCA="mlx5_0")
        spec = contract.build_launch_spec(env, sysfs_ib_dir=self.sysfs)
        self.assertFalse(spec.mounts.learn_enabled)
        self.assertNotIn("--learn-conversations", spec.argv)

    def test_rank1_default_flush_grace_exceeds_rank0_shutdown_grace(self):
        self.assertGreater(entrypoint.DEFAULT_RANK1_FLUSH_GRACE_S, entrypoint.DEFAULT_SHUTDOWN_GRACE_S)

    def test_rank0_forwards_second_sigterm_to_cut_drain_short(self):
        hits = self.tmp / "hits.txt"
        script = (
            "import signal, sys, time\n"
            f"p = {str(hits)!r}\n"
            "n = [0]\n"
            "def on_term(s, f):\n"
            "    n[0] += 1\n"
            "    open(p, 'w').write(str(n[0]))\n"
            "    if n[0] >= 2:\n"
            "        sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "while True: time.sleep(0.03)\n"
        )
        spec = self._spec_for_python_script(script, rank="0")
        signals: list[int] = []
        cfg = entrypoint.SupervisorConfig(
            shutdown_grace_s=5.0, memguard_enabled=False, memguard_interval_s=0.05,
            state_file=self.tmp / "state0_two.json",
        )

        def send_two():
            time.sleep(0.1)
            signals.append(signal.SIGTERM)
            time.sleep(0.15)
            signals.append(signal.SIGTERM)

        t = threading.Thread(target=send_two, daemon=True)
        t.start()
        rc = entrypoint.supervise(spec, cfg, install_signals=False, external_signals=signals, log=lambda _: None)
        t.join(timeout=2)
        self.assertEqual(rc, 0)
        self.assertEqual(hits.read_text(), "2")

    def test_rank1_escalation_preserves_actual_child_exit_code(self):
        script = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(42))\n"
            "while True: time.sleep(0.05)\n"
        )
        spec = self._spec_for_python_script(script, rank="1")
        signals: list[int] = []
        cfg = entrypoint.SupervisorConfig(
            rank1_flush_grace_s=0.15, rank1_kill_grace_s=1.0,
            memguard_enabled=False, memguard_interval_s=0.05,
            state_file=self.tmp / "state1_rc42.json",
        )

        def send_later():
            time.sleep(0.08)
            signals.append(signal.SIGTERM)

        t = threading.Thread(target=send_later, daemon=True)
        t.start()
        rc = entrypoint.supervise(
            spec, cfg, sync_fn=lambda: None, install_signals=False, external_signals=signals, log=lambda _: None
        )
        t.join(timeout=2)
        self.assertEqual(rc, 42)

    def test_group_teardown_kills_lingering_descendant_when_leader_exits(self):
        pid_file = self.tmp / "grandchild.pid"
        script = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
            f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
            "sys.exit(0)\n"
        )
        spec = self._spec_for_python_script(script, rank="0")
        cfg = entrypoint.SupervisorConfig(
            memguard_enabled=False, memguard_interval_s=0.05, state_file=self.tmp / "state_desc.json"
        )
        rc = entrypoint.supervise(spec, cfg, install_signals=False, log=lambda _: None)
        self.assertEqual(rc, 0)
        gpid = int(pid_file.read_text())
        time.sleep(0.1)
        with self.assertRaises(ProcessLookupError):
            os.kill(gpid, 0)

    def test_health_probe_redacts_secret_on_connection_error(self):
        ok, rep = health.probe(
            url="http://127.0.0.1:1",
            mode="ready",
            timeout=0.5,
            api_key="sk-never-leak-me-1234",
            env={"TENSORFOLD_API_KEY": "sk-never-leak-me-1234"},
        )
        self.assertFalse(ok)
        self.assertNotIn("sk-never-leak-me-1234", json.dumps(rep))


class HealthProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _serve(self, routes: dict[str, tuple[int, dict]]):
        received_auth = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                received_auth.append(self.headers.get("Authorization"))
                status, payload = routes.get(self.path, (404, {"error": "not found"}))
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        srv.daemon_threads = True
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}", received_auth

    def test_ready_and_capacity_probe_with_v136_log_and_redacted_secrets(self):
        log_file = self.tmp / "rank0.log"
        log_file.write_text(
            "info: kv pool: the engine's own and 3 more taking 5044 pages of 512 rows (15.76 GiB) as they need them\n"
        )
        srv, url, auths = self._serve({
            "/health": (200, {"status": "ok", "model": "GLM-5.3-Flash", "max_batch_size": 4, "warming": False}),
            "/v1/models": (200, {"data": [{
                "id": "GLM-5.3-Flash",
                "context_length": 1048576,
                "max_model_len": 1048576,
                "reasoning": {"supported_efforts": ["max", "high", "low", "none"]},
            }]}),
        })
        try:
            ok, report = health.probe(
                url=url,
                mode="ready",
                api_key="top-secret-token-xyz",
                expected_model="GLM-5.3-Flash",
                expected_lanes=4,
                expected_context=1048576,
                expected_efforts=("max", "high", "low", "none"),
                min_pool_tokens=health.POOL_FLOOR_4X1M,
                check_models=True,
                startup_log_path=str(log_file),
            )
            self.assertTrue(ok, report)
            self.assertEqual(report["pool_tokens"], 2_582_528)
            self.assertEqual(auths[0], "Bearer top-secret-token-xyz")
            self.assertNotIn("top-secret-token-xyz", json.dumps(report))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_draining_fails_readiness_but_passes_liveness(self):
        srv, url, _ = self._serve({"/health": (503, {"status": "draining", "requests_in_progress": 1})})
        try:
            ok_ready, rep_ready = health.probe(url=url, mode="ready")
            ok_live, rep_live = health.probe(url=url, mode="live")
            self.assertFalse(ok_ready, rep_ready)
            self.assertTrue(ok_live, rep_live)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_keyed_minimal_health_and_worker_state_probe(self):
        ok, rep = health.evaluate_health(200, {"status": "ok"}, expected_model="GLM-5.3-Flash", expected_lanes=4)
        self.assertTrue(ok, rep)

        state = self.tmp / "worker.json"
        state.write_text(json.dumps({
            "rank": 1, "world": 2, "pid": os.getpid(), "child_pid": os.getpid(), "phase": "running",
        }))
        self.assertTrue(health.check_worker_state(state, mode="ready", proc_ppid_fn=lambda _: os.getpid())[0])
        state.write_text(json.dumps({
            "rank": 1, "world": 2, "pid": os.getpid(), "child_pid": os.getpid(), "phase": "flushing",
        }))
        self.assertFalse(health.check_worker_state(state, mode="ready", proc_ppid_fn=lambda _: os.getpid())[0])
        self.assertTrue(health.check_worker_state(state, mode="live", proc_ppid_fn=lambda _: os.getpid())[0])

        # Rejects rank 0, dead supervisor, and reused/mismatched child ppid.
        state.write_text(json.dumps({"rank": 0, "world": 2, "pid": os.getpid(), "child_pid": os.getpid(), "phase": "running"}))
        self.assertFalse(health.check_worker_state(state, mode="ready", proc_ppid_fn=lambda _: os.getpid())[0])
        state.write_text(json.dumps({"rank": 1, "world": 2, "pid": 9999999, "child_pid": os.getpid(), "phase": "running"}))
        self.assertFalse(health.check_worker_state(state, mode="ready", pid_alive=lambda p: p == os.getpid())[0])
        state.write_text(json.dumps({"rank": 1, "world": 2, "pid": os.getpid(), "child_pid": os.getpid(), "phase": "running"}))
        self.assertFalse(health.check_worker_state(state, mode="ready", proc_ppid_fn=lambda _: 12345)[0])

    def test_probe_refuses_redirects_and_redacts_structured_model_field(self):
        srv, url, _ = self._serve({
            "/health": (200, {"status": "ok", "model": "leak-sk-secret-value-99", "max_batch_size": 4, "warming": False}),
        })
        try:
            ok, rep = health.probe(
                url=url, mode="ready", api_key="sk-secret-value-99", expected_model="GLM-5.3-Flash",
            )
            self.assertFalse(ok)
            self.assertNotIn("sk-secret-value-99", json.dumps(rep))
        finally:
            srv.shutdown()
            srv.server_close()

        # Verify nested JSON object keys containing secrets are redacted too
        redacted_nested = health._redact_obj(
            {"nested": {"key-sk-nested-42": "val-sk-nested-42"}},
            env=None,
            extra_secrets=("sk-nested-42",),
        )
        self.assertNotIn("sk-nested-42", json.dumps(redacted_nested))

        # Verify 302 redirect is refused without following.
        redirect_hits = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                redirect_hits.append(self.path)
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/steal")
                self.end_headers()

        rsrv = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        rsrv.daemon_threads = True
        threading.Thread(target=rsrv.serve_forever, daemon=True).start()
        try:
            ok, rep = health.probe(
                url=f"http://127.0.0.1:{rsrv.server_address[1]}", mode="ready", api_key="sk-no-redirect",
            )
            self.assertFalse(ok)
            self.assertEqual(redirect_hits, ["/health"])
            self.assertNotIn("sk-no-redirect", json.dumps(rep))
        finally:
            rsrv.shutdown()
            rsrv.server_close()

    def test_probe_uses_first_comma_separated_key_redacts_all_members_and_requires_models_context(self):
        # 1. evaluate_models fails when expected_context is given but entry omits context_length and max_model_len
        ok_missing, rep_missing = health.evaluate_models(
            200,
            {"data": [{"id": "GLM-5.3-Flash"}]},
            expected_model="GLM-5.3-Flash",
            expected_context=1_048_576,
        )
        self.assertFalse(ok_missing, rep_missing)
        self.assertIn("context_length", rep_missing["reason"])

        # 2. Comma-separated TENSORFOLD_API_KEY authenticates with first non-empty key and redacts all members
        srv, url, auths = self._serve({
            "/health": (200, {
                "status": "ok",
                "model": "echo-second-key-9999",
                "max_batch_size": 4,
                "warming": False,
            }),
        })
        try:
            ok, rep = health.probe(
                url=url,
                mode="ready",
                api_key=" , first-key-1111 , second-key-9999 ",
                expected_model="GLM-5.3-Flash",
            )
            self.assertFalse(ok)
            self.assertEqual(auths, ["Bearer first-key-1111"])
            dumped = json.dumps(rep)
            self.assertNotIn("first-key-1111", dumped)
            self.assertNotIn("second-key-9999", dumped)
        finally:
            srv.shutdown()
            srv.server_close()

        # 3. Short (<4 char) comma-separated key (e.g. "abc,another-valid-key") authenticates with "abc" and redacts it
        srv_short, url_short, auths_short = self._serve({
            "/health": (200, {
                "status": "ok",
                "model": "leaked abc and another-valid-key (but keep abcd intact)",
                "max_batch_size": 4,
                "warming": False,
            }),
        })
        try:
            ok_s, rep_s = health.probe(
                url=url_short,
                mode="ready",
                api_key="abc,another-valid-key",
                expected_model="GLM-5.3-Flash",
            )
            self.assertFalse(ok_s)
            self.assertEqual(auths_short, ["Bearer abc"])
            self.assertNotIn(" abc ", rep_s["model"])
            self.assertNotIn("another-valid-key", rep_s["model"])
            self.assertIn("abcd", rep_s["model"])
            self.assertIn("<redacted>", rep_s["model"])
        finally:
            srv_short.shutdown()
            srv_short.server_close()


if __name__ == "__main__":
    unittest.main()
