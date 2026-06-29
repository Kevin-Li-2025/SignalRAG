import unittest

from scripts.bge_m3_lora_utils import (
    LinearModuleInfo,
    build_lora_plan,
    select_lora_modules,
    target_modules_for_preset,
)


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool = True):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class FakeLinear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = object()


class FakeModel:
    def named_modules(self):
        modules = {
            "encoder.layer.0.attention.self.query": FakeLinear(8, 8),
            "encoder.layer.0.attention.self.key": FakeLinear(8, 8),
            "encoder.layer.0.attention.self.value": FakeLinear(8, 8),
            "encoder.layer.0.output.dense": FakeLinear(8, 32),
            "pooler.dense": FakeLinear(8, 8),
        }
        return modules.items()

    def named_parameters(self):
        return [
            ("encoder.layer.0.attention.self.query.weight", FakeParameter(64, False)),
            ("encoder.layer.0.attention.self.value.weight", FakeParameter(64, False)),
            ("pooler.dense.weight", FakeParameter(64, False)),
        ]


class BgeM3LoraUtilsTest(unittest.TestCase):
    def test_target_modules_for_preset_or_explicit_override(self):
        self.assertEqual(target_modules_for_preset("attention_qv"), ["query", "value"])
        self.assertEqual(target_modules_for_preset("attention_qv", "query,value"), ["query", "value"])
        self.assertEqual(target_modules_for_preset("all_linear"), "all-linear")

    def test_select_lora_modules_honors_suffix_and_exclusion(self):
        modules = [
            LinearModuleInfo("encoder.layer.0.attention.self.query", 8, 8, 64),
            LinearModuleInfo("encoder.layer.0.attention.self.value", 8, 8, 64),
            LinearModuleInfo("pooler.dense", 8, 8, 64),
        ]

        selected = select_lora_modules(
            modules,
            target_modules=["query", "value", "dense"],
            exclude_regex="pooler",
        )

        self.assertEqual([module.name for module in selected], [
            "encoder.layer.0.attention.self.query",
            "encoder.layer.0.attention.self.value",
        ])

    def test_build_lora_plan_estimates_trainable_parameters(self):
        plan = build_lora_plan(
            model=FakeModel(),
            target_preset="attention_qv",
            rank=4,
            alpha=8,
            dropout=0.05,
            exclude_regex="pooler",
        )

        self.assertEqual(plan["selected_module_count"], 2)
        self.assertEqual(plan["estimated_lora_trainable_parameters"], 128)
        self.assertEqual(plan["total_parameters"], 192)
        self.assertEqual(plan["trainable_parameters"], 0)
        self.assertGreater(plan["estimated_lora_trainable_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
