"""Fail-closed filesystem confinement and sanitized subprocess environments.

macOS uses sandbox-exec; Linux uses Landlock. Neither makes native GPU code
safe for a hostile multi-tenant service. Run production evaluations in disposable
containers/VMs with no credentials, restricted devices, and independent verifiers.
The explicit trusted-fixture exception has no OS sandbox and requires exact
repository mock proposal checks in both the evaluator and worker.
"""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import platform
import sys


STRICT = "strict"
TRUSTED_FIXTURE_COLAB = "trusted_fixture_colab"
COLAB_NVIDIA_DRIVER_LIBRARY = Path("/usr/lib64-nvidia")


def landlock_status() -> dict:
    if sys.platform != "linux":
        return {"available": None, "reason": "not applicable on this OS", "errno": None}
    if platform.machine() not in ("x86_64", "aarch64"):
        return {"available": False, "reason": "unsupported Landlock architecture", "errno": None}
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    abi = libc.syscall(444, 0, 0, 1)
    error = ctypes.get_errno() if abi < 1 else None
    return {"available": abi >= 1, "abi": abi if abi >= 1 else None, "errno": error,
            "reason": (f"Landlock ABI {abi}" if abi >= 1 else
                       f"Landlock unavailable: {errno.errorcode.get(error, error)}; kernel or host seccomp policy")}


def strict_isolation_unavailable_reason() -> str | None:
    if sys.platform == "linux":
        status = landlock_status()
        return status["reason"] if not status["available"] else None
    if sys.platform == "darwin":
        return None if Path("/usr/bin/sandbox-exec").exists() else "macOS sandbox-exec unavailable"
    return "filesystem confinement unsupported on this OS"


def isolation_report(mode: str) -> dict:
    if mode not in (STRICT, TRUSTED_FIXTURE_COLAB):
        raise ValueError(f"unknown isolation mode: {mode}")
    trusted = mode == TRUSTED_FIXTURE_COLAB
    return {
        "mode": mode, "explicit_opt_in": trusted,
        "trust_scope": ("exact repository mock fixtures and trusted framework baselines only" if trusted else
                        "AST-restricted source under required OS filesystem confinement"),
        "filesystem_confinement": "none" if trusted else "required",
        "landlock": landlock_status(), "production_sandbox": False,
        "warning": ("No OS sandbox: trusted fixtures only; not safe for arbitrary or replay source" if trusted else None),
    }


def worker_environment(directory: Path, isolation_mode: str = STRICT) -> dict[str, str]:
    if isolation_mode not in (STRICT, TRUSTED_FIXTURE_COLAB):
        raise ValueError(f"unknown isolation mode: {isolation_mode}")
    environment = {
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(directory), "TMPDIR": str(directory), "TMP": str(directory),
        "TEMP": str(directory), "XDG_CACHE_HOME": str(directory / "cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_DIR": str(directory / "triton"),
        "CUDA_CACHE_PATH": str(directory / "cuda-cache"),
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "USER": "agent-eval", "LOGNAME": "agent-eval",
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    # Colab exposes libcuda through this system-owned directory. Do not inherit
    # arbitrary LD_LIBRARY_PATH entries from the notebook or user environment.
    if sys.platform == "linux" and COLAB_NVIDIA_DRIVER_LIBRARY.is_dir():
        environment["LD_LIBRARY_PATH"] = str(COLAB_NVIDIA_DRIVER_LIBRARY)
    return environment


def worker_command(directory: Path, isolation_mode: str = STRICT) -> list[str]:
    bootstrap = f"import sys; sys.path.insert(0, {str(directory)!r}); import candidate_worker; candidate_worker.main()"
    command = [sys.executable, "-I", "-B", "-c", bootstrap]
    if isolation_mode == TRUSTED_FIXTURE_COLAB:
        return command
    if isolation_mode != STRICT:
        raise ValueError(f"unknown isolation mode: {isolation_mode}")
    if sys.platform == "darwin":
        executable = Path("/usr/bin/sandbox-exec")
        if not executable.exists():
            raise RuntimeError("sandbox-exec unavailable; refusing unconfined evaluation")
        import json

        quote = json.dumps
        profile = (
            "(version 1)(allow default)(deny network*)(deny file-write*)"
            f"(allow file-write* (subpath {quote(str(directory))}) (literal \"/dev/null\"))"
            f"(deny file-read* (subpath {quote(str(Path.home()))}))"
            f"(allow file-read* (subpath {quote(str(directory))})"
            f" (subpath {quote(sys.prefix)}) (subpath {quote(sys.base_prefix)}))"
        )
        return [str(executable), "-p", profile, *command]
    if sys.platform != "linux":
        raise RuntimeError("filesystem sandbox unavailable on this OS; refusing unconfined evaluation")
    return command


def restrict_linux_filesystem(directory: Path) -> str:
    if sys.platform != "linux":
        return "macOS sandbox-exec; repository writes and network denied"
    status = landlock_status()
    if not status["available"]:
        raise RuntimeError(status["reason"] + "; refusing unconfined evaluation")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = status["abi"]
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attribute = Ruleset(handled)
    descriptor = libc.syscall(444, ctypes.byref(attribute), ctypes.sizeof(attribute), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Landlock create_ruleset failed")
    try:
        paths = {Path(root): 13 for root in (sys.prefix, sys.base_prefix, "/usr", "/lib", "/lib64",
                                            "/bin", "/sbin", "/sys", "/proc/driver/nvidia")}
        paths[COLAB_NVIDIA_DRIVER_LIBRARY] = 13
        paths.update({Path(root): 4 for root in ("/etc/ld.so.cache", "/etc/localtime", "/etc/os-release",
                                               "/etc/passwd", "/etc/group", "/etc/nsswitch.conf",
                                               "/proc/cpuinfo", "/proc/meminfo", "/proc/self/maps",
                                               "/proc/self/status", "/proc/self/stat")})
        paths[Path("/dev")] = 12
        paths[Path("/dev/null")] = 6
        for path in Path("/dev").glob("nvidia*"):
            paths[path] = 15 if path.is_dir() else 6
        paths[directory] = handled
        for path, access in paths.items():
            if not path.exists():
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access & handled, parent)
                if libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"Landlock rule failed: {path}")
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, descriptor, 0) != 0:
            raise OSError(ctypes.get_errno() or errno.EPERM, "Landlock restrict_self failed")
    finally:
        os.close(descriptor)
    return f"Linux Landlock ABI {abi}; writes restricted to trial directory"
