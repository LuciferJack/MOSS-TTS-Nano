"""Gradient projection helpers for teacher updates with preservation constraints."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectionReport:
    iterations: int
    original_minimum_dot: float
    minimum_dot: float
    retained_norm_ratio: float
    original_dots: tuple[float, ...] = ()
    dots: tuple[float, ...] = ()


def project_teacher_gradient(
    teacher: torch.Tensor,
    protectors: list[torch.Tensor],
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    minimum_retained_ratio: float = 0.05,
) -> tuple[torch.Tensor, ProjectionReport]:
    """Project ``teacher`` until it has non-negative dot product with every protector.

    The zero vector is always feasible, but an update that loses nearly all of its
    norm is rejected because it cannot provide a useful teacher signal.
    """
    if teacher.ndim != 1 or not protectors or any(item.shape != teacher.shape for item in protectors):
        raise ValueError("teacher and non-empty protectors must be same-shaped vectors")
    if not torch.isfinite(teacher).all() or any(not torch.isfinite(item).all() for item in protectors):
        raise ValueError("gradient vectors must be finite")
    original_norm = teacher.norm()
    if original_norm <= tolerance:
        raise ValueError("teacher gradient norm is zero")
    denominators = [item.dot(item) for item in protectors]
    if any(value <= tolerance for value in denominators):
        raise ValueError("protector gradient norm is zero")

    original_minimum_dot = float(torch.stack([teacher.dot(item) for item in protectors]).min())
    projected = teacher.clone()
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        changed = False
        for protector, denominator in zip(protectors, denominators):
            dot = projected.dot(protector)
            if dot < -tolerance:
                projected = projected - dot / denominator * protector
                changed = True
        dots = torch.stack([projected.dot(item) for item in protectors])
        if not changed or float(dots.min()) >= -tolerance:
            break
    else:
        raise RuntimeError("protected gradient projection did not converge")

    dots = torch.stack([projected.dot(item) for item in protectors])
    minimum_dot = float(dots.min())
    if minimum_dot < -tolerance:
        raise RuntimeError(f"projection left a negative preservation dot product: {minimum_dot}")
    retained_ratio = float(projected.norm() / original_norm)
    if retained_ratio < minimum_retained_ratio:
        raise RuntimeError(f"teacher signal collapsed during projection: retained={retained_ratio:.6f}")
    return projected, ProjectionReport(
        iterations, original_minimum_dot, minimum_dot, retained_ratio,
        tuple(float(teacher.dot(item)) for item in protectors),
        tuple(float(projected.dot(item)) for item in protectors),
    )
