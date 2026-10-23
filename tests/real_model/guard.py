"""Read-only guard for a known earlier run; never terminates processes."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def prior_run_blocker(filename):
    if not filename or not Path(filename).is_file():
        return None
    state=json.loads(Path(filename).read_text())
    if state.get('status') != 'running':
        return None
    pid=state.get('pid')
    if type(pid) is not int or pid<=0:
        return 'Prior run has no valid PID; cannot establish that it has stopped.'
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        # A recent heartbeat can be stale immediately after Ctrl+C. No wait required.
        return None
    except PermissionError:
        stamp=state.get('heartbeat_at')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(stamp)).total_seconds() if stamp else None
        suffix=f' Last heartbeat {age:.1f} seconds ago.' if age is not None else ''
        return f'Cannot inspect prior PID {pid} from this process; its status is unknown.{suffix} Run the test in regular Terminal after confirming the earlier process stopped.'
    # PID exists. Distinguish an exited zombie or reused PID when readable.
    try:
        import psutil
        process=psutil.Process(pid)
        if process.status()==psutil.STATUS_ZOMBIE:
            return None
        started=state.get('started_at')
        if started and process.create_time()>datetime.fromisoformat(started).timestamp()+1:
            return None  # PID was reused for a process created after this run started.
    except (ImportError, OSError):
        pass
    except Exception as exc:
        # psutil AccessDenied/NoSuchProcess are platform-specific exceptions.
        if type(exc).__name__ in ('NoSuchProcess','ZombieProcess'):
            return None
        if type(exc).__name__!='AccessDenied':
            raise
    return f'Prior run PID {pid} still exists. Wait for it to exit or stop that run before loading another 8B model.'
