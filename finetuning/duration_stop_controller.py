"""Frozen-model, formula-scoped duration/stop arbitration prototype.

The controller never changes MOSS weights.  It learns a conservative stop
window from teacher boundaries and only overrides a continuing decoder at the
window's safe upper edge.  Native EOS always wins.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence


FORMULA = re.compile(r"(?:[A-Z][a-z]?|[₀-₉0-9()·])+" )
ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og".split()
)
FORMULA_TOKEN = re.compile(r"[A-Z][a-z]?|[₀-₉0-9()·]")


@dataclass(frozen=True)
class StopTrace:
    case_id: str
    canonical_text: str
    spoken_text: str
    teacher_boundary_frame: int
    slot_logits: tuple[float, ...]
    end_logits: tuple[float, ...]
    max_frames: int


@dataclass(frozen=True)
class StopWindow:
    earliest_frame: int
    safe_stop_frame: int
    confidence: float
    formula_scoped: bool


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def validate_trace(trace: StopTrace) -> None:
    if not trace.case_id or not trace.canonical_text or not trace.spoken_text:
        raise ValueError("trace requires non-empty identity and exact text fields")
    if not 0 < trace.teacher_boundary_frame < trace.max_frames:
        raise ValueError("teacher boundary must be inside the generation frame limit")
    if len(trace.slot_logits) != len(trace.end_logits) or not trace.slot_logits:
        raise ValueError("slot/end trajectories must be non-empty and aligned")
    if len(trace.slot_logits) < trace.teacher_boundary_frame:
        raise ValueError("logit trajectory ends before the teacher boundary")
    if not _finite(trace.slot_logits) or not _finite(trace.end_logits):
        raise ValueError("logit trajectories must be finite")


def is_formula_scoped(canonical_text: str) -> bool:
    matches = FORMULA.findall(canonical_text)
    for candidate in matches:
        tokens = FORMULA_TOKEN.findall(candidate)
        elements = [token for token in tokens if token[0].isalpha()]
        valid_chemistry = "".join(tokens) == candidate and elements and all(
            element in ELEMENTS for element in elements
        )
        has_structure = any(char in "₀₁₂₃₄₅₆₇₈₉()·" for char in candidate)
        explicit_context = any(marker in canonical_text for marker in ("化学式", "分子式", "水合"))
        # Plain ASCII digits are ambiguous with versions (SDK2); accept those
        # only under explicit chemistry context. Unicode subscripts/brackets/
        # hydrate dots are independently strong formula syntax.
        if valid_chemistry and (has_structure or (explicit_context and any(char.isdigit() for char in candidate))):
            return True
    return False


def text_features(canonical: str, spoken: str) -> tuple[float, ...]:
    formula = max(FORMULA.findall(canonical), key=len, default="")
    return (
        float(len(formula)), float(sum(char.isupper() for char in formula)),
        float(sum(char.isdigit() or char in "₀₁₂₃₄₅₆₇₈₉" for char in formula)),
        float(formula.count("·")), float(formula.count("(")),
        float(len(spoken)), float(len(canonical)),
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    # Fixed scales prevent spoken length from drowning formula structure.
    scales = (8.0, 4.0, 4.0, 1.0, 1.0, 20.0, 20.0)
    return math.sqrt(sum(((a - b) / scale) ** 2 for a, b, scale in zip(left, right, scales)))


class ConservativeStopController:
    def __init__(self, traces: Iterable[StopTrace], *, minimum_confidence: float = 0.6):
        self.traces = tuple(traces)
        if len(self.traces) < 3:
            raise ValueError("at least three independent formula teachers are required")
        if len({trace.case_id for trace in self.traces}) != len(self.traces):
            raise ValueError("teacher case IDs must be unique")
        for trace in self.traces:
            validate_trace(trace)
            if not is_formula_scoped(trace.canonical_text):
                raise ValueError("controller teachers must be formula-scoped")
        self.minimum_confidence = minimum_confidence
        self._features = tuple(text_features(t.canonical_text, t.spoken_text) for t in self.traces)
        # Inner leave-one-out residuals make the stop edge one-sided: it is
        # never earlier than any observed teacher under-prediction.
        errors, distances = [], []
        for index, trace in enumerate(self.traces):
            candidates = [(_distance(self._features[index], f), j)
                          for j, f in enumerate(self._features) if j != index]
            distance, nearest = min(candidates)
            errors.append(max(0, trace.teacher_boundary_frame - self.traces[nearest].teacher_boundary_frame))
            distances.append(distance)
        self._underprediction_guard = max(errors)
        self._distance_limit = max(distances)

    def window(self, canonical_text: str, spoken_text: str, *, max_frames: int) -> StopWindow:
        scoped = is_formula_scoped(canonical_text)
        if not scoped:
            return StopWindow(0, max_frames, 0.0, False)
        features = text_features(canonical_text, spoken_text)
        ranked = sorted((_distance(features, known), i) for i, known in enumerate(self._features))
        distance, nearest = ranked[0]
        predicted = self.traces[nearest].teacher_boundary_frame
        confidence = max(0.0, 1.0 - distance / max(self._distance_limit, 1e-9))
        safe = min(
            max_frames - 1,
            max(predicted + self._underprediction_guard,
                max(trace.teacher_boundary_frame for trace in self.traces)),
        )
        earliest = max(1, min(t.teacher_boundary_frame for t in self.traces))
        return StopWindow(earliest, safe, confidence, True)

    def decide(self, *, frame_index: int, slot_logit: float, end_logit: float,
               window: StopWindow) -> str:
        if not _finite((slot_logit, end_logit)):
            raise ValueError("runtime logits must be finite")
        if end_logit >= slot_logit:
            return "native_eos"
        if not window.formula_scoped or window.confidence < self.minimum_confidence:
            return "continue"
        if frame_index < window.earliest_frame:
            return "continue"
        if frame_index >= window.safe_stop_frame:
            return "controller_eos"
        return "continue"


def leave_one_formula_out(traces: Sequence[StopTrace]) -> list[dict]:
    """Return auditable predictions; never converts a failed fold into PASS."""
    results = []
    for index, heldout in enumerate(traces):
        controller = ConservativeStopController(t for i, t in enumerate(traces) if i != index)
        window = controller.window(heldout.canonical_text, heldout.spoken_text,
                                   max_frames=heldout.max_frames)
        results.append({
            "case_id": heldout.case_id,
            "teacher_boundary_frame": heldout.teacher_boundary_frame,
            "safe_stop_frame": window.safe_stop_frame,
            "confidence": window.confidence,
            "early_cut": window.safe_stop_frame < heldout.teacher_boundary_frame,
            "before_cap": window.safe_stop_frame < heldout.max_frames,
        })
    return results
