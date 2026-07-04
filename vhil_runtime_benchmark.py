#!/usr/bin/env python3
"""Collect vHIL runtime evidence from service health endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.request import urlopen


def _get_path(obj: dict, path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fetch_json(url: str, timeout_s: float) -> dict:
    with urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _wait_healthy(urls: dict[str, str], timeout_s: float, request_timeout_s: float) -> dict[str, dict]:
    deadline = time.time() + timeout_s
    last_error: dict[str, str] = {}
    while time.time() < deadline:
        snapshots: dict[str, dict] = {}
        all_ok = True
        for name, url in urls.items():
            try:
                payload = _fetch_json(url, request_timeout_s)
                snapshots[name] = payload
                all_ok = all_ok and bool(payload.get("ok"))
            except Exception as exc:
                last_error[name] = str(exc)
                all_ok = False
        if all_ok:
            return snapshots
        time.sleep(0.5)
    raise RuntimeError(f"services did not become healthy before timeout: {last_error}")


def _delta(final: dict, initial: dict, path: str) -> int:
    return int(_get_path(final, path, 0) or 0) - int(_get_path(initial, path, 0) or 0)


def _loss_pct(sent: int, received: int) -> float:
    if sent <= 0:
        return 0.0
    return max(0.0, 100.0 * (sent - received) / sent)


def _summarize(samples: list[dict], initial: dict, final: dict, args: argparse.Namespace) -> dict:
    physim_sensor_tx = _delta(final["physim"], initial["physim"], "state.sensor_packets")
    controller_sensor_rx = _delta(final["controller"], initial["controller"], "state.sensor_packets")
    controller_command_tx = _delta(final["controller"], initial["controller"], "state.command_packets")
    physim_command_rx = _delta(final["physim"], initial["physim"], "state.command_packets")

    controller_metrics = _get_path(final["controller"], "state.metrics", {}) or {}
    physim_metrics = _get_path(final["physim"], "state.metrics", {}) or {}
    rtt = controller_metrics.get("closed_loop_rtt_ms", {})
    sensor_jitter = controller_metrics.get("sensor_interarrival_ms", {})
    controller_loop_jitter = controller_metrics.get("loop_jitter_ms", {})
    physim_loop_jitter = physim_metrics.get("loop_jitter_ms", {})

    sensor_loss_pct = _loss_pct(physim_sensor_tx, controller_sensor_rx)
    command_loss_pct = _loss_pct(controller_command_tx, physim_command_rx)
    checks = {
        "controller_healthy": bool(final["controller"].get("ok")),
        "physim_healthy": bool(final["physim"].get("ok")),
        "closed_loop_rtt_samples": int(rtt.get("count", 0) or 0) > 0,
        "sensor_loss_within_limit": sensor_loss_pct <= float(args.max_loss_pct),
        "command_loss_within_limit": command_loss_pct <= float(args.max_loss_pct),
        "mean_rtt_within_limit": float(rtt.get("mean", 0.0) or 0.0) <= float(args.max_mean_rtt_ms),
    }

    return {
        "duration_s": float(args.duration),
        "sample_interval_s": float(args.sample_interval),
        "target_hz": float(args.hz),
        "packet_deltas": {
            "physim_sensor_tx": physim_sensor_tx,
            "controller_sensor_rx": controller_sensor_rx,
            "controller_command_tx": controller_command_tx,
            "physim_command_rx": physim_command_rx,
        },
        "loss": {
            "sensor_link_pct": sensor_loss_pct,
            "command_link_pct": command_loss_pct,
            "controller_sequence": controller_metrics.get("sensor_sequence", {}),
            "physim_sequence": physim_metrics.get("command_sequence", {}),
        },
        "timing": {
            "closed_loop_rtt_ms": rtt,
            "sensor_interarrival_ms": sensor_jitter,
            "controller_loop_jitter_ms": controller_loop_jitter,
            "physim_loop_jitter_ms": physim_loop_jitter,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _plot(samples: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    elapsed = [sample["elapsed_s"] for sample in samples]
    physim_sensor = [_get_path(sample, "physim.state.sensor_packets", 0) for sample in samples]
    controller_sensor = [_get_path(sample, "controller.state.sensor_packets", 0) for sample in samples]
    controller_command = [_get_path(sample, "controller.state.command_packets", 0) for sample in samples]
    physim_command = [_get_path(sample, "physim.state.command_packets", 0) for sample in samples]
    rtt_mean = [_get_path(sample, "controller.state.metrics.closed_loop_rtt_ms.mean", 0.0) for sample in samples]
    rtt_max = [_get_path(sample, "controller.state.metrics.closed_loop_rtt_ms.max", 0.0) for sample in samples]
    sensor_jitter = [_get_path(sample, "controller.state.metrics.sensor_interarrival_ms.std", 0.0) for sample in samples]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(elapsed, physim_sensor, label="physim sensor TX")
    axes[0].plot(elapsed, controller_sensor, label="controller sensor RX")
    axes[0].plot(elapsed, controller_command, label="controller command TX")
    axes[0].plot(elapsed, physim_command, label="physim command RX")
    axes[0].set_ylabel("packets")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(elapsed, rtt_mean, label="RTT mean")
    axes[1].plot(elapsed, rtt_max, label="RTT max")
    axes[1].set_ylabel("RTT (ms)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(elapsed, sensor_jitter, label="sensor interarrival std")
    axes[2].set_xlabel("elapsed (s)")
    axes[2].set_ylabel("jitter (ms)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect vHIL 100 Hz runtime metrics")
    parser.add_argument("--duration", type=float, default=float(os.getenv("VHIL_BENCHMARK_DURATION", "600")))
    parser.add_argument("--sample-interval", type=float, default=float(os.getenv("VHIL_BENCHMARK_SAMPLE_INTERVAL", "1")))
    parser.add_argument("--hz", type=float, default=float(os.getenv("VHIL_RATE_HZ", "100")))
    parser.add_argument("--ready-timeout", type=float, default=float(os.getenv("VHIL_READY_TIMEOUT", "60")))
    parser.add_argument("--request-timeout", type=float, default=2.0)
    parser.add_argument("--max-loss-pct", type=float, default=float(os.getenv("VHIL_MAX_LOSS_PCT", "1.0")))
    parser.add_argument("--max-mean-rtt-ms", type=float, default=float(os.getenv("VHIL_MAX_MEAN_RTT_MS", "100.0")))
    parser.add_argument("--controller-url", default=os.getenv("VHIL_CONTROLLER_HEALTH_URL", "http://127.0.0.1:8082/health"))
    parser.add_argument("--physim-url", default=os.getenv("VHIL_PHYSIM_HEALTH_URL", "http://127.0.0.1:8081/health"))
    parser.add_argument("--output-dir", default=os.getenv("VHIL_BENCHMARK_OUTPUT_DIR", "runs/vhil_runtime"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    urls = {"controller": args.controller_url, "physim": args.physim_url}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_ready = _wait_healthy(urls, float(args.ready_timeout), float(args.request_timeout))
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    initial = {
        "controller": initial_ready["controller"],
        "physim": initial_ready["physim"],
    }
    samples: list[dict] = []

    while (time.perf_counter() - started_perf) < float(args.duration):
        sample_started = time.perf_counter()
        sample = {
            "elapsed_s": sample_started - started_perf,
            "controller": _fetch_json(args.controller_url, float(args.request_timeout)),
            "physim": _fetch_json(args.physim_url, float(args.request_timeout)),
        }
        samples.append(sample)
        sleep_s = float(args.sample_interval) - (time.perf_counter() - sample_started)
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    final = {
        "controller": _fetch_json(args.controller_url, float(args.request_timeout)),
        "physim": _fetch_json(args.physim_url, float(args.request_timeout)),
    }
    summary = _summarize(samples, initial, final, args)
    finished_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"vhil_runtime_{stamp}.json"
    png_path = output_dir / f"vhil_runtime_{stamp}.png"

    report = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "urls": urls,
        "initial": initial,
        "final": final,
        "summary": summary,
        "samples": samples,
        "artifacts": {
            "json": str(json_path),
            "png": str(png_path),
        },
    }

    try:
        _plot(samples, png_path)
    except Exception as exc:
        report["artifacts"]["png_error"] = str(exc)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": bool(summary["passed"]), "summary": summary, "artifacts": report["artifacts"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
