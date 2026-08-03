import os
import time
from contextlib import contextmanager
import errno

from utils.logger import logger


DATA_ACCESS_LOCK_NAME = "data_access"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


@contextmanager
def run_lock(name: str, lock_dir: str = "logs/locks", timeout_sec: float = 0.0, exit_code: int = 0):
    """Prevent overlapping cron runs via atomic PID lockfile.

    The context value is ``True`` when this call had to wait for a live peer
    before acquiring the lock, otherwise ``False``.  Existing callers ignore
    the value and retain their previous behavior.
    """
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{name}.lock")

    start = time.time()
    acquired = False
    waited_for_live_peer = False
    try:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
                finally:
                    os.close(fd)
                acquired = True
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

                existing_pid = None
                try:
                    with open(lock_path, "r") as f:
                        s = (f.read() or "").strip()
                        existing_pid = int(s) if s.isdigit() else None
                except Exception:
                    existing_pid = None

                if existing_pid and _pid_is_alive(existing_pid):
                    waited_for_live_peer = True
                    if timeout_sec and (time.time() - start) < float(timeout_sec):
                        time.sleep(0.2)
                        continue
                    logger.warning(f"[Lock] Another run active (pid={existing_pid}). Exiting.")
                    raise SystemExit(exit_code)

                try:
                    os.remove(lock_path)
                except Exception:
                    if timeout_sec and (time.time() - start) < float(timeout_sec):
                        time.sleep(0.2)
                        continue
                    logger.warning(f"[Lock] Could not clear stale lock. Exiting.")
                    raise SystemExit(exit_code)

        yield waited_for_live_peer
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except Exception:
                pass


def data_access_lock(
    timeout_sec: float,
    exit_code: int = os.EX_TEMPFAIL,
    lock_dir: str = "logs/locks",
):
    """Serialize DB writers and multi-query readers on one shared lock."""
    return run_lock(
        DATA_ACCESS_LOCK_NAME,
        lock_dir=lock_dir,
        timeout_sec=timeout_sec,
        exit_code=exit_code,
    )


def wait_for_lock_release(
    name: str,
    timeout_sec: float,
    exit_code: int = os.EX_TEMPFAIL,
    lock_dir: str = "logs/locks",
) -> None:
    """Wait for a live legacy/exclusive lock without claiming ownership."""
    lock_path = os.path.join(lock_dir, f"{name}.lock")
    start = time.time()
    while True:
        try:
            with open(lock_path, "r") as handle:
                raw_pid = (handle.read() or "").strip()
        except FileNotFoundError:
            return
        except OSError:
            raw_pid = ""

        pid = int(raw_pid) if raw_pid.isdigit() else None
        if pid and _pid_is_alive(pid):
            if timeout_sec and (time.time() - start) < float(timeout_sec):
                time.sleep(0.2)
                continue
            logger.warning(f"[Lock] Another run active (pid={pid}). Exiting.")
            raise SystemExit(exit_code)

        try:
            os.remove(lock_path)
        except FileNotFoundError:
            return
        except OSError:
            if timeout_sec and (time.time() - start) < float(timeout_sec):
                time.sleep(0.2)
                continue
            logger.warning(f"[Lock] Could not clear stale lock. Exiting.")
            raise SystemExit(exit_code)


@contextmanager
def stable_data_read_lock(
    timeout_sec: float,
    exit_code: int = os.EX_TEMPFAIL,
    lock_dir: str = "logs/locks",
):
    """Hold a stable DB view across multi-query readers.

    First drain an updater that may have started before ``data_access`` was
    introduced, then share one bounded timeout with the current writer lock.
    New updaters claim ``update_data`` before ``data_access``, so the hand-off
    is race-safe without making a reader look like a completed updater.
    """
    started = time.time()
    wait_for_lock_release(
        "update_data",
        timeout_sec=timeout_sec,
        exit_code=exit_code,
        lock_dir=lock_dir,
    )
    elapsed = time.time() - started
    remaining = max(0.0, float(timeout_sec) - elapsed)
    with data_access_lock(
        timeout_sec=remaining,
        exit_code=exit_code,
        lock_dir=lock_dir,
    ):
        yield
