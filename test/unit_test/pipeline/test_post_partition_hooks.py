import unittest

from neuronx_distributed.trainer import PostPartitionHooks


def function(a, b):
    a = a * b
    return a


class TestPostPartitionHooks(unittest.TestCase):
    def test_execute_post_partition_hook(self):
        hooks = PostPartitionHooks()
        inputs = [5,6]
        hooks.register_post_partition_hook(function, inputs)
        output = hooks.execute_all_hooks()
        expected_result = 30
        assert output[0] == expected_result

    def test_register_post_partition_hooks(self):
        hooks = PostPartitionHooks()
        inputs = [5,6]
        hooks.register_post_partition_hook(function, inputs)
        hooks.register_post_partition_hook(function, inputs)

        expected_len_hooks = 2
        assert len(hooks.hooks) == expected_len_hooks
        # Execute hooks and validate that hooks get cleared
        hooks.execute_all_hooks()
        assert len(hooks.hooks) == 0


if __name__ == "__main__":
    unittest.main()
