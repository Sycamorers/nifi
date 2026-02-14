import faulthandler
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional



def enable_faulthandler_signal() -> None:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass

    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
        except Exception:
            pass



def _read_meminfo() -> Dict[str, float]:
    mem_total_kb = None
    mem_avail_kb = None
    try:
        with Path("/proc/meminfo").open("r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = float(line.split()[1])
    except Exception:
        pass

    return {
        "mem_total_gb": (mem_total_kb / (1024.0 * 1024.0)) if mem_total_kb else None,
        "mem_available_gb": (mem_avail_kb / (1024.0 * 1024.0)) if mem_avail_kb else None,
    }



def snapshot_system() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
    }

    try:
        l1, l5, l15 = os.getloadavg()
        payload["loadavg"] = {"1m": l1, "5m": l5, "15m": l15}
    except Exception:
        payload["loadavg"] = None

    payload.update(_read_meminfo())

    try:
        ps = subprocess.run(
            ["ps", "-p", str(os.getpid()), "-o", "%cpu,%mem,rss,vsz,etime,comm", "--no-headers"],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["self_ps"] = ps.stdout.strip()
    except Exception:
        payload["self_ps"] = None

    return payload



def snapshot_gpu() -> Dict[str, Any]:
    # Try pynvml first.
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            out.append(
                {
                    "index": i,
                    "name": name,
                    "gpu_util": float(util.gpu),
                    "mem_util": float(util.memory),
                    "mem_used_gb": mem.used / (1024 ** 3),
                    "mem_total_gb": mem.total / (1024 ** 3),
                }
            )
        pynvml.nvmlShutdown()
        return {"backend": "pynvml", "gpus": out}
    except Exception:
        pass

    # Fallback to nvidia-smi.
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        gpus = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 7:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_util": float(parts[2]),
                    "mem_util": float(parts[3]),
                    "mem_used_mb": float(parts[4]),
                    "mem_total_mb": float(parts[5]),
                    "temp_c": float(parts[6]),
                }
            )
        return {"backend": "nvidia-smi", "gpus": gpus}
    except Exception as e:
        return {"backend": "none", "error": str(e)}



def process_tree_snapshot(root_pid: Optional[int] = None) -> Dict[str, Any]:
    root_pid = int(root_pid if root_pid is not None else os.getpid())

    try:
        cmd = ["ps", "-eo", "pid,ppid,stat,%cpu,%mem,etime,command", "--no-headers"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        rows = []
        for ln in proc.stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(maxsplit=6)
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "pid": int(parts[0]),
                    "ppid": int(parts[1]),
                    "stat": parts[2],
                    "cpu": parts[3],
                    "mem": parts[4],
                    "etime": parts[5],
                    "command": parts[6],
                }
            )

        by_ppid: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_ppid.setdefault(r["ppid"], []).append(r)

        descendants = []
        queue = [root_pid]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            for child in by_ppid.get(cur, []):
                descendants.append(child)
                queue.append(child["pid"])

        return {
            "root_pid": root_pid,
            "descendants": descendants,
        }
    except Exception as e:
        return {"root_pid": root_pid, "error": str(e)}



def list_recent_outputs(out_dir: Path, top_k: int = 50) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []

    files = []
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
            files.append(
                {
                    "path": str(p),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                }
            )
        except Exception:
            continue

    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files[:top_k]



def dump_stack_traces(pid: int) -> bool:
    try:
        if pid == os.getpid():
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            return True
        if hasattr(signal, "SIGUSR1"):
            os.kill(pid, signal.SIGUSR1)
            return True
    except Exception:
        return False
    return False



def collect_diagnostics(stage: str, out_dir: Path, last_substep: str, pid: Optional[int] = None) -> Dict[str, Any]:
    target_pid = int(pid if pid is not None else os.getpid())
    payload = {
        "stage": stage,
        "last_successful_substep": last_substep,
        "target_pid": target_pid,
        "system": snapshot_system(),
        "gpu": snapshot_gpu(),
        "process_tree": process_tree_snapshot(root_pid=target_pid),
        "recent_outputs": list_recent_outputs(Path(out_dir), top_k=50),
    }
    return payload



def write_diagnostics(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
