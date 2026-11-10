"""Use the actual supported SDK to resolve and translate the external release profile, without hosts."""
import importlib.util
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec('sparkrun'), 'Install the pinned SparkRun test dependency')
class SdkContractTests(unittest.TestCase):
    def test_external_profile_and_adjacent_mod_resolve_without_mutable_registry(self):
        from sparkrun.core.recipe import Recipe
        from sparkrun.core import mods
        adapter = load('external_tensorfold_test', ROOT / 'plugins/tensorfold/__init__.py')
        launcher = load('release_launcher_test', ROOT / 'recipes/glm53-mia/sparkrun/mods/glm53-native/serve.py')
        profile = ROOT / 'recipes/glm53-mia/sparkrun/profiles/glm53-mia-2026.11.07-agent-1m.yaml'

        class Registry:
            def get_registry(self, *args):
                raise AssertionError('Unexpected mutable-registry fallback')

        with patch('subprocess.Popen', side_effect=AssertionError('Unexpected process launch')), \
             patch('socket.create_connection', side_effect=AssertionError('Unexpected network connection')):
            recipe = Recipe.load(profile, resolve=False)
            runtime = adapter.TensorFoldRuntime()
            self.assertEqual(runtime.validate_recipe(recipe), [])
            mod = mods._resolve_local(recipe.mods[0], recipe, Registry())
            self.assertEqual(Path(mod.source_path), ROOT / 'recipes/glm53-mia/sparkrun/mods/glm53-native')
            context = adapter._FABRIC_MASTER.set('192.0.2.1')
            try:
                for rank in (0, 1):
                    command = runtime.generate_node_command(recipe, {}, head_ip='192.0.2.10',
                                                            num_nodes=2, node_rank=rank, init_port=25000)
                    env = dict(word.split('=', 1) for word in shlex.split(command)
                               if '=' in word and not word.startswith('--'))
                    env.update(recipe.env)
                    env['NCCL_IB_HCA'] = 'mlx5_0,mlx5_1'
                    argv, mapped = launcher.launch(env)
                    self.assertEqual(mapped['NCCL_COMM_ID'], '192.0.2.1:24999')
                    self.assertEqual(mapped['TF_GLM_RANK'], str(rank))
                    self.assertEqual(argv[argv.index('--learn-gib') + 1], '64')
            finally:
                adapter._FABRIC_MASTER.reset(context)


if __name__ == '__main__':
    unittest.main()
