import os
import sys


MODES = {"server", "benchmark"}


def _exec(command):
    os.execvp(sys.executable, [sys.executable, *command])


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] not in MODES:
        os.execvp(args[0], args)

    mode = args[0] if args else os.getenv("USV_MODE", "server")
    mode_args = args[1:] if args and args[0] in MODES else []

    if mode == "server":
        _exec(["visualization_server.py", *mode_args])
    if mode == "benchmark":
        _exec(["benchmark_engine.py", *mode_args])

    raise SystemExit(f"Unsupported USV_MODE: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())