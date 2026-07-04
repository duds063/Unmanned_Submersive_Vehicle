#!/usr/bin/env python3
"""Small runtime metric helpers for vHIL services."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, value: float) -> None:
        x = float(value)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self.min_value = x if self.min_value is None else min(self.min_value, x)
        self.max_value = x if self.max_value is None else max(self.max_value, x)

    def snapshot(self) -> dict:
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "count": self.count,
            "mean": self.mean,
            "std": math.sqrt(max(0.0, variance)),
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class SequenceTracker:
    last_sequence: Optional[int] = None
    missing: int = 0
    out_of_order: int = 0

    def update(self, sequence: Optional[int]) -> None:
        if sequence is None:
            return

        seq = int(sequence)
        if seq <= 0:
            return

        if self.last_sequence is None:
            self.last_sequence = seq
            return

        if seq > self.last_sequence + 1:
            self.missing += seq - self.last_sequence - 1
        elif seq <= self.last_sequence:
            self.out_of_order += 1

        if seq > self.last_sequence:
            self.last_sequence = seq

    def snapshot(self) -> dict:
        received_span = int(self.last_sequence or 0)
        loss_pct = (100.0 * self.missing / received_span) if received_span > 0 else 0.0
        return {
            "last_sequence": self.last_sequence,
            "missing": self.missing,
            "out_of_order": self.out_of_order,
            "loss_pct": loss_pct,
        }
