"""Off-host contracts for the released native image's launch and admission receipts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'recipes/glm53-mia/sparkrun/mods/glm53-native'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load('native_serve', MOD / 'serve.py')
capacity = load('native_capacity', MOD / 'capacity.py')
readiness = load('native_readiness', MOD / 'readiness.py')
renderer = load('profile_renderer', ROOT / 'tools/render_sparkrun.py')
verify = load('runtime_verify', MOD / 'verify.py')


class NativeProfileTests(unittest.TestCase):
    def env(self, rank='0'):
        return {'RANK': rank, 'FABRIC_MASTER': '192.0.2.1', 'MASTER_PORT': '25000',
                'NCCL_IB_HCA': 'mlx5_0,mlx5_1', 'CONTEXT': '1048576', 'MAX_TOKENS': '32768',
                'MODEL_PATH': '/weights/target', 'DRAFTER': '/weights/draft', 'SERVED_NAME': 'GLM-5.3-Flash',
                'HOST': '0.0.0.0', 'PORT': '8888',
                'GLM_LEARN_DIR': '/cache/huggingface/spark-inference/learn/test',
                'GLM_LEARN_GIB': '64', 'GLM_LEARN_MIN_FREE_GIB': '50', 'GLM_LEARN_DAILY_GIB': '128'}

    def test_both_ranks_gate_watch_port_and_use_identical_serving_flags(self):
        first, env0 = serve.launch(self.env())
        second, env1 = serve.launch(self.env('1'))
        self.assertEqual(first, second)
        self.assertEqual(env0['NCCL_COMM_ID'], '192.0.2.1:24999')
        self.assertEqual(env1['TF_GLM_RANK'], '1')
        self.assertEqual(env0['NCCL_IB_HCA'], 'mlx5_0,mlx5_1')
        for flag, value in {'--top-k': '20', '--parallel': '4', '--prompt-cache-gib': '6',
                            '--learn-gib': '64', '--learn-min-free-gib': '50',
                            '--learn-daily-gib': '128', '--reasoning-effort': 'xhigh'}.items():
            self.assertEqual(first[first.index(flag) + 1], value)
        self.assertIn('--thinking', first)
        self.assertIn('--learn-conversations', first)
        self.assertNotIn('--spill-gib', first)

    def test_unsafe_placement_and_store_overrides_refuse_before_spawn(self):
        for changes in ({'RANK': '2'}, {'MASTER_PORT': '1'}, {'FABRIC_MASTER': '::1'},
                        {'CONTEXT': '524288'}, {'GLM_LEARN_DIR': '/cache/huggingface/tensorfold-spill'},
                        {'GLM_LEARN_DIR': '/cache/huggingface/spark-inference/learn/../escape'},
                        {'EXTRA_ARGS': '--context 8'}, {'GLM_LEARN_GIB': '0'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                serve.launch(dict(self.env(), **changes))

    def log(self, pages=6147, cache='4.1', disk=True):
        text = f"[tensorfold] CUDA GLM: the engine's own and 3 more taking {pages} pages of 512 rows\n"
        text += f'info: prompt cache: {cache} GiB of device pages and 0.0 GiB of pinned host memory a rank\n'
        if disk:
            text += "info: glm: conversation states on disk in /cache/learn/identity (0 states, 0 MiB of 65536 MiB; 2 ranks' parts)\n"
        return text

    def test_capacity_distinguishes_configured_context_funded_cache_and_store(self):
        result = capacity.check(self.log())
        self.assertEqual(result['aggregate_pool_tokens'], 3144704)
        self.assertEqual(result['minimum_part_pages'], 3071)
        self.assertEqual(result['learned_store_location'], 'rank0')
        self.assertEqual(result['durability_status'], 'unknown')
        for text in (self.log(pages=2049), self.log(pages=5044), self.log(cache='0.4'), self.log(disk=False), self.log() + self.log()):
            with self.subTest(text=text), self.assertRaises(ValueError):
                capacity.check(text)
        with self.assertRaisesRegex(ValueError, 'unexpected directory'):
            capacity.check(self.log(), learn_dir='/cache/huggingface/spark-inference/learn/another-release')

    def test_sharded_health_requires_a_full_window_per_part(self):
        health = {'status': 'ok', 'warming': False, 'model': 'GLM-5.3-Flash', 'max_batch_size': 4,
                  'kv_shard': {'parts': 2, 'page_tokens': 512, 'part_pages': 3073, 'per_part': [{}, {}]}}
        readiness.check_health(health, 'GLM-5.3-Flash', 4, 1048576, require_kv_shard=True)
        short = copy.deepcopy(health)
        short['kv_shard']['part_pages'] = 512
        with self.assertRaisesRegex(RuntimeError, 'cannot hold'):
            readiness.check_health(short, 'GLM-5.3-Flash', 4, 1048576, require_kv_shard=True)
        del short['kv_shard']
        with self.assertRaises(RuntimeError):
            readiness.check_health(short, 'GLM-5.3-Flash', 4, 1048576, require_kv_shard=True)

    def test_profile_and_template_are_generated_from_the_published_release(self):
        release = ROOT / 'recipes/glm53-mia/releases/glm53-mia-2026.11.07.json'
        for path, text in renderer.outputs(release).items():
            self.assertEqual(path.read_text(), text, str(path))
        template = (MOD / 'chat_template.jinja').read_text()
        self.assertNotIn(renderer.BASE_LINE, template)
        self.assertEqual(template.count(renderer.EFFORT_LINE), 1)
        # This pinned template has a dangling think opener but no switch name; unmodified discovery omits none.
        self.assertIn('<think>', template)
        self.assertNotIn('enable_thinking', template)
        self.assertTrue(verify.verify(MOD))

    def test_changed_mod_is_refused(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'mod'
            shutil.copytree(MOD, target, ignore=shutil.ignore_patterns('__pycache__'))
            (target / 'serve.py').write_text('changed')
            with self.assertRaises(ValueError):
                verify.verify(target)

    def test_parked_memory_counts_only_pages_above_resting_level(self):
        text = ('Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: 524288\n      high_min: 262144\n'
                '  vm stats threshold: 1\n    cpu: 1\n      count: 524288\n      high_min: 262144\n'
                '  next_zone_field\n      count: 999999999\n')
        with tempfile.TemporaryDirectory() as directory:
            memory, zones = Path(directory) / 'meminfo', Path(directory) / 'zoneinfo'
            memory.write_text('MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\n')
            zones.write_text(text)
            self.assertEqual(serve.available_gib(memory, zones, 4096), 4)
            memory.write_text('MemTotal: 3145728 kB\nMemAvailable: 2097152 kB\n')
            self.assertEqual(serve.available_gib(memory, zones, 4096), 3)
            memory.write_text('MemFree: 1 kB\n')
            with self.assertRaisesRegex(ValueError, 'MemAvailable'):
                serve.available_gib(memory, zones, 4096)
        for bad in (text.replace('count: 524288', 'count: -1'), text.replace('high_min: 262144', 'high_min: nope')):
            self.assertEqual(serve.parked_bytes(bad, 4096), 0)

    def test_supervisor_requires_continuous_low_memory_and_stops_on_sampler_loss(self):
        clock = [0]

        class Child:
            def poll(self):
                return None

            def wait(self, timeout):
                clock[0] += timeout
                raise subprocess.TimeoutExpired('unit-fixture', timeout)

        with mock.patch.object(serve.subprocess, 'Popen', return_value=Child()), \
                mock.patch.object(serve.signal, 'signal'), mock.patch.object(serve.time, 'monotonic', side_effect=lambda: clock[0]), \
                mock.patch.object(serve, 'available_gib', side_effect=[2, 2, 3, 2, 2, 2]), \
                mock.patch.object(serve, 'stop') as stop:
            self.assertEqual(serve.supervise(['unit-fixture'], {'RANK': '1'}), 73)
            self.assertEqual(clock[0], 10)
            stop.assert_called_once()
            self.assertEqual(stop.call_args.kwargs['grace'], 10)
        with mock.patch.object(serve.subprocess, 'Popen', return_value=Child()), \
                mock.patch.object(serve.signal, 'signal'), \
                mock.patch.object(serve, 'available_gib', side_effect=OSError('unit-fixture')), \
                mock.patch.object(serve, 'stop'):
            self.assertEqual(serve.supervise(['unit-fixture'], {'RANK': '0'}), 72)

    def test_stop_kills_only_private_synthetic_process_group(self):
        child = subprocess.Popen([sys.executable, '-u', '-c',
                'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready"); time.sleep(60)'],
                start_new_session=True, stdout=subprocess.PIPE, text=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            serve.stop(child, grace=0.05)
            self.assertEqual(child.returncode, -signal.SIGKILL)
            self.assertIsNone(unrelated.poll())
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            child.stdout.close()
            unrelated.terminate()
            unrelated.wait()


if __name__ == '__main__':
    unittest.main()
