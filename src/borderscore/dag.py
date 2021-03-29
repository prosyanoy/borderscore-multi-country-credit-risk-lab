"""A tiny dependency-free DAG runner (the same task table also feeds the Airflow DAG)."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: Callable
    deps: tuple[str, ...] = field(default_factory=tuple)


def topological_order(tasks: list[TaskSpec]) -> list[TaskSpec]:
    by_name = {t.name: t for t in tasks}
    if len(by_name) != len(tasks):
        raise ValueError("duplicate task names")
    for t in tasks:
        for d in t.deps:
            if d not in by_name:
                raise ValueError(f"task {t.name!r} depends on unknown task {d!r}")
    order, state = [], {}

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"cycle detected at task {name!r}")
        state[name] = 1
        for d in by_name[name].deps:
            visit(d)
        state[name] = 2
        order.append(by_name[name])

    for t in tasks:
        visit(t.name)
    return order


def run_dag(tasks: list[TaskSpec], ctx) -> dict[str, dict]:
    """Run tasks in dependency order. A failed task marks its descendants ``upstream_failed``."""
    results: dict[str, dict] = {}
    for t in topological_order(tasks):
        if any(results[d]["status"] != "success" for d in t.deps):
            results[t.name] = {"status": "upstream_failed", "seconds": 0.0}
            continue
        start = time.time()
        try:
            t.fn(ctx)
            results[t.name] = {"status": "success", "seconds": time.time() - start}
        except Exception:
            results[t.name] = {"status": "failed", "seconds": time.time() - start, "error": traceback.format_exc()}
    return results
