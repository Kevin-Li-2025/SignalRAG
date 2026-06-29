import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO

from scripts.run_flagembedding_m3_no_ddp import (
    parse_wrapper_args,
    patch_flagembedding_m3_runner_for_lora,
    patch_torch_distributed_tensor_for_peft,
    patch_torch_distributed_for_single_process,
)


class DummyDistributed:
    def __init__(self, *, available=True, initialized=False):
        self.available = available
        self.initialized = initialized
        self.rank_calls = 0
        self.world_size_calls = 0

    def is_available(self):
        return self.available

    def is_initialized(self):
        return self.initialized

    def get_rank(self, *args, **kwargs):
        self.rank_calls += 1
        if not self.initialized:
            raise RuntimeError("process group not initialized")
        return 7

    def get_world_size(self, *args, **kwargs):
        self.world_size_calls += 1
        if not self.initialized:
            raise RuntimeError("process group not initialized")
        return 3


class RunFlagEmbeddingM3NoDdpTest(unittest.TestCase):
    def test_patch_returns_single_process_values_when_uninitialized(self):
        dist = DummyDistributed(initialized=False)

        patch_torch_distributed_for_single_process(dist)

        self.assertEqual(dist.get_rank(), 0)
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.rank_calls, 0)
        self.assertEqual(dist.world_size_calls, 0)

    def test_patch_delegates_when_process_group_is_initialized(self):
        dist = DummyDistributed(initialized=True)

        patch_torch_distributed_for_single_process(dist)

        self.assertEqual(dist.get_rank(), 7)
        self.assertEqual(dist.get_world_size(), 3)
        self.assertEqual(dist.rank_calls, 1)
        self.assertEqual(dist.world_size_calls, 1)

    def test_patch_adds_distributed_tensor_placeholder_for_peft(self):
        class DistWithoutTensor:
            pass

        dist = DistWithoutTensor()
        patch_torch_distributed_tensor_for_peft(dist)

        self.assertTrue(hasattr(dist, "tensor"))
        self.assertTrue(hasattr(dist.tensor, "DTensor"))

    def test_parse_wrapper_args_strips_lora_options(self):
        wrapper_args, forwarded = parse_wrapper_args([
            "--lora-enable",
            "--lora-r",
            "8",
            "--model_name_or_path",
            "BAAI/bge-m3",
        ])

        self.assertTrue(wrapper_args.lora_enable)
        self.assertEqual(wrapper_args.lora_r, 8)
        self.assertEqual(forwarded, ["--model_name_or_path", "BAAI/bge-m3"])

    def test_lora_patch_wraps_runner_get_model(self):
        class FakeRunner:
            @staticmethod
            def get_model(*_args, **_kwargs):
                return {"model": FakeEncoder()}

        class FakeEncoder:
            def named_modules(self):
                return [
                    ("encoder.layer.0.attention.self.query", FakeLinear(8, 8)),
                    ("encoder.layer.0.attention.self.value", FakeLinear(8, 8)),
                ]

            def named_parameters(self):
                return [("p", FakeParameter(128, False))]

        class FakeLinear:
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = object()

        class FakeParameter:
            def __init__(self, count, requires_grad):
                self.count = count
                self.requires_grad = requires_grad

            def numel(self):
                return self.count

        class FakeTaskType:
            FEATURE_EXTRACTION = "FEATURE_EXTRACTION"

        seen = {}

        def fake_config(**kwargs):
            seen["config"] = kwargs
            return kwargs

        def fake_get_peft_model(model, config):
            seen["wrapped"] = True
            model.config_seen = config
            return model

        patch_flagembedding_m3_runner_for_lora(
            FakeRunner,
            wrapper_args=Namespace(
                lora_target_preset="attention_qv",
                lora_target_modules="",
                lora_include_regex="",
                lora_exclude_regex="",
                lora_r=4,
                lora_alpha=8,
                lora_dropout=0.05,
                lora_bias="none",
                lora_task_type="FEATURE_EXTRACTION",
                lora_report_json="",
            ),
            lora_config_cls=fake_config,
            task_type_cls=FakeTaskType,
            get_peft_model_fn=fake_get_peft_model,
        )

        with redirect_stdout(StringIO()):
            payload = FakeRunner.get_model()

        self.assertTrue(seen["wrapped"])
        self.assertEqual(seen["config"]["target_modules"], ["query", "value"])
        self.assertEqual(payload["model"].config_seen["r"], 4)


if __name__ == "__main__":
    unittest.main()
