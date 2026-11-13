"""Console/file logging for experiment runs plus tools for diagnosing jobs that look idle:
a heartbeat thread reporting the current stage, and on-demand Python stack dumps."""

import faulthandler
import logging
import os
import resource
import signal
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
LEVELS = (logging.INFO, VERBOSE, logging.DEBUG)
ROOT_LOGGER = "src"
FORMAT = "%(asctime)s %(levelname).1s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    """Line-buffered stdout/stderr, `src.*` loggers to stdout at INFO / VERBOSE / DEBUG
    (verbosity 0 / 1 / 2+), and faulthandler (SIGUSR2 dumps all thread stacks to stderr)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)

    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(LEVELS[min(max(verbosity, 0), len(LEVELS) - 1)])
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    logger.addHandler(handler)

    faulthandler.enable()
    if hasattr(signal, "SIGUSR2"):
        faulthandler.register(signal.SIGUSR2, all_threads=True)


def add_log_file(log_dir: Path) -> Path:
    """Also writes all `src.*` log records to `log_dir/<timestamp>_<SLURM job id or pid>.log`."""
    log_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{job}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    logging.getLogger(ROOT_LOGGER).addHandler(handler)
    return path


@contextmanager
def timed(logger: logging.Logger, message: str, level: int = logging.INFO) -> Generator[None]:
    """Logs `message ...` before and `message done in X s` after a potentially blocking step."""
    logger.log(level, "%s ...", message)
    start = time.perf_counter()
    yield
    logger.log(level, "%s done in %.1f s", message, time.perf_counter() - start)


def dump_stacks_periodically(minutes: float) -> None:
    if minutes > 0:
        faulthandler.dump_traceback_later(minutes * 60, repeat=True)


# ---------- current stage (read by the heartbeat) ----------
# Setting a key clears every deeper key not set in the same call (a new method drops the
# previous trial/seed/epoch).
STAGE_KEYS = ("dataset", "fold", "phase", "method", "trial", "seed", "epoch")
_stage: dict[str, Any] = {}
_stage_updated = time.monotonic()
_stage_lock = threading.Lock()


def set_stage(**fields: Any) -> None:
    global _stage_updated
    unknown = set(fields) - set(STAGE_KEYS)
    if unknown:
        raise ValueError(f"Unknown stage keys {unknown}")
    depth = min(STAGE_KEYS.index(k) for k in fields)
    with _stage_lock:
        for key in STAGE_KEYS[depth:]:
            if key in fields and fields[key] is not None:
                _stage[key] = fields[key]
            else:
                _stage.pop(key, None)
        _stage_updated = time.monotonic()


def current_stage() -> tuple[str, float]:
    """(stage description, seconds since the last stage update)."""
    with _stage_lock:
        text = " ".join(f"{k}={_stage[k]}" for k in STAGE_KEYS if k in _stage) or "startup"
        return text, time.monotonic() - _stage_updated


def _memory_summary() -> str:
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        text = f"rss {rss:.0f} MB"
    except (OSError, ValueError):
        text = f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        allocated = torch.cuda.memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        text += f", cuda {allocated:.0f}/{reserved:.0f} MB allocated/reserved"
    return text


def start_heartbeat(minutes: float) -> None:
    """Logs the current stage every `minutes` from a daemon thread. It keeps running while the
    main thread is blocked in C code, so an unchanging stage with a growing "last update"
    indicates a hang rather than slow progress."""
    if minutes <= 0:
        return

    def beat() -> None:
        start = time.monotonic()
        while True:
            time.sleep(minutes * 60)
            stage, idle = current_stage()
            log.info(
                "[heartbeat] %s | last update %.0f s ago | running %.1f h | %s",
                stage,
                idle,
                (time.monotonic() - start) / 3600,
                _memory_summary(),
            )

    threading.Thread(target=beat, name="heartbeat", daemon=True).start()
