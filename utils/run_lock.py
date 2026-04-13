import os
import time
from contextlib import contextmanager
import errno

from utils.logger import logger


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
    """Prevent overlapping cron runs via atomic PID lockfile."""
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{name}.lock")

    start = time.time()
    acquired = False
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

        yield
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except Exception:
                pass
