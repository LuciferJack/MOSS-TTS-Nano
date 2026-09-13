from __future__ import annotations

import unittest
from argparse import Namespace

import torch

from finetuning.protected_gradient import project_teacher_gradient
from finetuning.sft import pcgrad_backward, resolve_objective_loss_weights


class _Accelerator:
    @staticmethod
    def backward(loss):
        loss.backward()


class ProtectedGradientTests(unittest.TestCase):
    def test_teacher_and_protector_use_separate_channel_objectives(self):
        teacher, protector = resolve_objective_loss_weights(
            Namespace(
                channelwise_loss_weight="1,0",
                protect_channelwise_loss_weight="0,1",
            ),
            n_heads=9,
        )
        self.assertEqual(teacher, [1.0] + [0.0] * 8)
        self.assertEqual(protector, [0.0] + [0.125] * 8)

    def test_projects_conflicts_and_preserves_compatible_component(self):
        teacher = torch.tensor([-2.0, -1.0, 3.0])
        protectors = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]
        projected, report = project_teacher_gradient(teacher, protectors)
        self.assertTrue(all(float(projected.dot(item)) >= 0 for item in protectors))
        self.assertAlmostEqual(float(projected[2]), 3.0)
        self.assertGreater(report.retained_norm_ratio, 0.5)

    def test_keeps_compatible_gradient_unchanged(self):
        teacher = torch.tensor([1.0, 2.0])
        projected, report = project_teacher_gradient(teacher, [torch.tensor([1.0, 0.0])])
        self.assertTrue(torch.equal(projected, teacher))
        self.assertEqual(report.iterations, 1)

    def test_rejects_collapsed_teacher_signal(self):
        with self.assertRaisesRegex(RuntimeError, "collapsed"):
            project_teacher_gradient(torch.tensor([-1.0, 0.0]), [torch.tensor([1.0, 0.0])])

    def test_rejects_non_finite_input(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            project_teacher_gradient(torch.tensor([float("nan")]), [torch.tensor([1.0])])

    def test_training_backward_uses_protector_only_as_constraint(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        model = torch.nn.ParameterList([parameter])
        teacher_loss = (-parameter[0] + parameter[1])
        protector_loss = parameter[0]
        report = pcgrad_backward(
            accelerator=_Accelerator(), model=model,
            teacher_loss=teacher_loss, protector_loss=protector_loss,
        )
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([0.0, 1.0])))
        self.assertGreaterEqual(report.minimum_dot, 0.0)

    def test_training_backward_does_not_apply_protector_only_parameters(self):
        teacher_parameter = torch.nn.Parameter(torch.tensor(0.0))
        protector_parameter = torch.nn.Parameter(torch.tensor(0.0))
        model = torch.nn.ParameterList([teacher_parameter, protector_parameter])
        teacher_loss = teacher_parameter
        protector_loss = teacher_parameter + 7.0 * protector_parameter
        pcgrad_backward(
            accelerator=_Accelerator(), model=model,
            teacher_loss=teacher_loss, protector_loss=protector_loss,
        )
        self.assertEqual(float(teacher_parameter.grad), 1.0)
        self.assertIsNone(protector_parameter.grad)

    def test_training_backward_preserves_existing_optimizer_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        model = torch.nn.ParameterList([parameter])
        parameter.grad = torch.tensor([3.0, 4.0])
        pcgrad_backward(
            accelerator=_Accelerator(), model=model,
            teacher_loss=(-parameter[0] + parameter[1]),
            protector_loss=parameter[0],
        )
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([3.0, 5.0])))


if __name__ == "__main__":
    unittest.main()
