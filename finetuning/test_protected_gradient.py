from __future__ import annotations

import unittest

import torch

from finetuning.protected_gradient import project_teacher_gradient
from finetuning.sft import pcgrad_backward


class _Accelerator:
    @staticmethod
    def backward(loss):
        loss.backward()


class ProtectedGradientTests(unittest.TestCase):
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

    def test_training_backward_combines_protector_and_projected_teacher(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        model = torch.nn.ParameterList([parameter])
        teacher_loss = (-parameter[0] + parameter[1])
        protector_loss = parameter[0]
        report = pcgrad_backward(
            accelerator=_Accelerator(), model=model,
            teacher_loss=teacher_loss, protector_loss=protector_loss,
        )
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([1.0, 1.0])))
        self.assertGreaterEqual(report.minimum_dot, 0.0)


if __name__ == "__main__":
    unittest.main()
