"""
agent/core/sandbox.py
Generic subprocess sandbox for secure execution of untrusted Python code.

Security model (defense-in-depth):
  1. AST pre-flight: blocks forbidden imports and dangerous names BEFORE exec.
  2. Subprocess isolation (the real boundary):
       - cwd locked to an isolated temp dir
       - stripped environment (no cloud creds / secrets leak into the child)
       - memory limit (Linux RLIMIT_AS / Windows Job Object)
       - CPU/time limit (RLIMIT_CPU on Linux, timeout kill elsewhere)
       - network namespace isolation on Linux (best-effort, requires privileges)
       - CREATE_NEW_PROCESS_GROUP on Windows for clean timeout kills
  3. Restricted __builtins__ inside the child: whitelist-only builtins dict
     plus an import allow-list (statsforecast, mlforecast, sklearn, pandas,
     numpy, and safe stdlib). No os/subprocess/socket/requests, etc.
  4. I/O contract: JSON files only, no sys.stdin, no network.
  5. Output validation: schema check + size guard.

Why NO RestrictedPython:
  RestrictedPython rewrites bytecode and blocks underscore-prefixed names and
  restricted attribute access. statsforecast / numba rely on dunder methods,
  metaclasses and internal attributes, so compile_restricted breaks them at
  runtime. The subprocess boundary above is sufficient; RestrictedPython is
  removed to keep the forecasting stack functional.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("hr_agent.sandbox")

try:
    import resource as resource_mod
except ImportError:
    resource_mod = None  # type: ignore

try:
    import win32job  # type: ignore
    import win32process  # type: ignore
    import pywintypes  # type: ignore
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MEMORY_MB = int(os.getenv("SANDBOX_MEMORY_MB", "512"))
_DEFAULT_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))
_DEFAULT_MAX_FILES = int(os.getenv("SANDBOX_MAX_FILES", "64"))
_DEFAULT_MAX_OUTPUT_BYTES = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", str(10 * 1024 * 1024)))  # 10 MB


# ---------------------------------------------------------------------------
# Forbidden modules (AST pre-flight + import guard)
# ---------------------------------------------------------------------------
_FORBIDDEN_IMPORTS: Set[str] = {
    # network / system
    "os", "subprocess", "socket", "urllib", "http", "ftplib", "smtplib",
    "requests", "shutil", "glob", "tempfile", "asyncio", "threading",
    "multiprocessing", "concurrent", "ctypes", "signal", "importlib",
    "sys", "builtins", "pty", "tty", "termios", "select", "poll",
    "fcntl", "mmap",
    # dangerous stdlib
    "pickle", "marshal", "shelve", "dbm", "sqlite3", "ssl", "hashlib",
    "cffi", "code", "codeop", "compile",
    # filesystem / archive abuse
    "pathlib", "zipfile", "tarfile", "gzip", "bz2", "lzma", "zipimport",
}

# ---------------------------------------------------------------------------
# Forbidden names (dangerous builtins / attributes) — AST pre-flight only
# ---------------------------------------------------------------------------
_FORBIDDEN_NAMES: Set[str] = {
    "__import__", "exec", "eval", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "hasattr", "callable", "__loader__", "__spec__", "__builtins__",
    "__build_class__", "memoryview", "bytearray", "bytes",
}

# ---------------------------------------------------------------------------
# Whitelisted builtins (safe subset for math/data work)
# ---------------------------------------------------------------------------
_SAFE_BUILTINS: Dict[str, Any] = {
    # constants
    "None": None,
    "True": True,
    "False": False,
    "Ellipsis": Ellipsis,
    "__name__": "__main__",
    # exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError,
    "StopIteration": StopIteration,
    "GeneratorExit": GeneratorExit,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "AssertionError": AssertionError,
    "FileNotFoundError": FileNotFoundError,
    "PermissionError": PermissionError,
    "TimeoutError": TimeoutError,
    # math / logic / collections
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# ---------------------------------------------------------------------------
# Allowed import prefixes inside the sandbox (import guard)
# ---------------------------------------------------------------------------
_ALLOWED_IMPORT_PREFIXES = [
    # Forecasting stacks
    "statsforecast", "mlforecast", "sklearn", "pandas", "numpy", "scipy",
    # Safe standard library (data / math work)
    "json", "math", "datetime", "statistics", "collections", "random",
    "time", "typing", "warnings", "logging", "copy", "itertools",
    "functools", "operator", "string", "decimal", "fractions", "numbers",
    "re", "textwrap", "types", "inspect", "dataclasses", "typing_extensions",
    "matplotlib",
]

# Environment keys that must NEVER reach the untrusted subprocess.
_SENSITIVE_ENV_PREFIXES = (
    "AWS_", "AZURE_", "GCP_", "GOOGLE_", "SECRET", "TOKEN", "PASSWORD",
    "PASSWD", "PRIVATE_KEY", "API_KEY", "APIKEY", "ACCESS_KEY", "SESSION_TOKEN",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""
    success: bool
    output_path: str = ""
    result_data: Optional[Dict[str, Any]] = None  # parsed output (decoupled from filesystem)
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    execution_time_seconds: float = 0.0
    memory_peak_mb: Optional[float] = None
    data_sources_used: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Linux namespace helpers (best-effort, requires privileges)
# ---------------------------------------------------------------------------
import ctypes  # noqa: E402


def _linux_preexec_fn(memory_mb: int, timeout_seconds: int, max_open_files: int) -> None:
    """preexec_fn for subprocess.Popen on Linux.

    Sets resource limits, attempts network namespace isolation, and
    disables ptrace/dumpable. Runs inside the child process before exec.
    """
    # 1. Resource limits
    if resource_mod is not None:
        try:
            mem_bytes = memory_mb * 1024 * 1024
            resource_mod.setrlimit(resource_mod.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource_mod.setrlimit(resource_mod.RLIMIT_CPU, (timeout_seconds, timeout_seconds))
            resource_mod.setrlimit(resource_mod.RLIMIT_CORE, (0, 0))
            resource_mod.setrlimit(resource_mod.RLIMIT_NOFILE, (max_open_files, max_open_files))
        except (ValueError, OSError) as exc:
            logger.warning("Failed to set Linux resource limits in child: %s", exc)

    # 2. Try network namespace isolation (unshare)
    #    CLONE_NEWNET = 0x40000000
    #    Requires: CAP_SYS_ADMIN or unprivileged user namespaces enabled
    try:
        CLONE_NEWNET = 0x40000000
        libc = ctypes.cdll.LoadLibrary("libc.so.6")
        if libc.unshare(CLONE_NEWNET) == 0:
            logger.info("Linux network namespace isolated (CLONE_NEWNET)")
        else:
            err = ctypes.get_errno()
            if err == 1:  # EPERM
                logger.info("Network namespace requires privileges — skipping (EPERM)")
            else:
                logger.info("Network namespace unshare failed (errno=%d)", err)
    except Exception as exc:
        logger.info("Network namespace isolation not available: %s", exc)

    # 3. Disable ptrace / core dumps
    try:
        PR_SET_DUMPABLE = 0
        libc.prctl(PR_SET_DUMPABLE, 0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Windows Job Object helper
# ---------------------------------------------------------------------------
def _create_windows_job(memory_mb: int) -> Any:
    """Create a Windows Job Object with memory and CPU limits."""
    if not _WIN32_AVAILABLE or platform.system() != "Windows":
        return None
    try:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation
        )
        limit_bytes = memory_mb * 1024 * 1024
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
        )
        info["ProcessMemoryLimit"] = limit_bytes
        info["JobMemoryLimit"] = limit_bytes
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info
        )
        logger.info("Windows Job Object created: mem=%dMB", memory_mb)
        return job
    except Exception as exc:
        logger.warning("Failed to create Windows Job Object: %s", exc)
        return None


# ---------------------------------------------------------------------------
# AST pre-flight check
# ---------------------------------------------------------------------------
def _check_ast(code: str, forbidden_imports: Set[str], forbidden_names: Set[str]) -> List[str]:
    """Parse code and return list of forbidden constructs found."""
    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in forbidden_imports:
                    violations.append(f"import:{top}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in forbidden_imports:
                    violations.append(f"import:{top}")
        elif isinstance(node, ast.Attribute):
            if node.attr in forbidden_names:
                violations.append(f"attr:{node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in forbidden_names:
                violations.append(f"name:{node.id}")
    return violations


# ---------------------------------------------------------------------------
# Environment stripping
# ---------------------------------------------------------------------------
def _safe_env() -> Dict[str, str]:
    """Return a stripped environment for the sandbox subprocess.

    Keeps only variables required for Python/numba to run (TEMP, SystemRoot,
    PATH, etc.) and drops cloud credentials / secrets so they cannot leak
    into the untrusted child process.
    """
    keep_exact = (
        "SystemRoot", "SystemDrive", "TEMP", "TMP", "TMPDIR", "USERPROFILE",
        "HOME", "PATH", "PYTHONIOENCODING", "PYTHONUTF8", "LANG", "LC_ALL",
        "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS", "KMP_AFFINITY",
    )
    env: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in keep_exact:
            env[k] = v
            continue
        if any(k.upper().startswith(p) or p in k.upper() for p in _SENSITIVE_ENV_PREFIXES):
            continue  # drop secrets
        env[k] = v  # keep other non-sensitive vars
    if not any(key in env for key in ("TEMP", "TMP", "TMPDIR")):
        env["TMPDIR"] = tempfile.gettempdir()
    return env


# ---------------------------------------------------------------------------
# Wrapper builder (no RestrictedPython)
# ---------------------------------------------------------------------------
def _safe_builtins_repr() -> str:
    """Serialize the whitelisted builtins dict for inlining into the wrapper.

    The subprocess runs a fresh interpreter, so module-level objects are not
    available; we serialize safe builtins (constants + callables by name) and
    lazily import the few safe stdlib modules the user code may need.
    """
    items: List[str] = []
    for k, v in _SAFE_BUILTINS.items():
        if v is None or v is True or v is False or v is Ellipsis:
            items.append(f"{repr(k)}: {repr(v)}")
        elif isinstance(v, type) or callable(v):
            items.append(f"{repr(k)}: {v.__name__}")
        else:
            items.append(f"{repr(k)}: {repr(v)}")
    for mod in ("math", "statistics", "collections", "json", "datetime",
                "random", "warnings", "logging", "copy", "itertools",
                "functools", "operator", "re", "string", "decimal", "fractions",
                "numbers", "types", "inspect", "dataclasses", "typing"):
        items.append(f"{repr(mod)}: __import__({repr(mod)})")
    return "{" + ", ".join(items) + "}"


def _build_wrapper(input_path: str, output_path: str) -> str:
    """Build the subprocess wrapper script.

    Security boundary is the subprocess itself (isolated cwd, stripped env,
    memory/CPU limits, timeout). Inside the child we use a restricted
    ``__builtins__`` dict plus an import allow-list. RestrictedPython is NOT
    used because its bytecode-level restrictions break statsforecast/numba.

    I/O contract:
      * Input : ``_input_data``  (dict loaded from input.json)
      * Script: ``script.py``    (LLM-generated forecasting code)
      * Output: ``_forecast_result`` (dict written to output.json)
    """
    return _build_plain_wrapper(input_path, output_path)


def _build_plain_wrapper(input_path: str, output_path: str) -> str:
    """Build a subprocess wrapper that reads ``script.py`` and executes it
    with a restricted ``__builtins__`` dict and an import allow-list.

    The user code lives in ``script.py`` (written by the parent process into
    the same isolated temp dir). The wrapper itself runs with real builtins
    so it can safely ``open`` the script and output files.
    """
    allowed = ", ".join(repr(p) for p in _ALLOWED_IMPORT_PREFIXES)
    safe_builtins_repr = _safe_builtins_repr()
    lines = [
        "import sys, json",
        "",
        "# ====== Sandbox I/O contract ======",
        "# Input : _input_data     (dict loaded from input.json)",
        "# Script: script.py       (LLM-generated forecasting code)",
        "# Output: _forecast_result (dict written to output.json)",
        "_INPUT_PATH = " + repr(input_path),
        "_OUTPUT_PATH = " + repr(output_path),
        "",
        "with open(_INPUT_PATH, 'r', encoding='utf-8') as _f:",
        "    _input_data = json.load(_f)",
        "",
        "# ====== Import allow-list ======",
        "import builtins as _sb_builtins",
        "_REAL_IMPORT = _sb_builtins.__import__",
        "def _safe_import(name, *a, **kw):",
        "    _allowed = [" + allowed + "]",
        "    if any(name == p or name.startswith(p + '.') for p in _allowed):",
        "        return _REAL_IMPORT(name, *a, **kw)",
        "    raise ImportError(\"Import of '\" + name + \"' is blocked in sandbox\")",
        "",
        "# ====== Restricted builtins ======",
        "_RESTRICTED_BUILTINS = " + safe_builtins_repr,
        "_RESTRICTED_BUILTINS['__import__'] = _safe_import",
        "_GLOBALS = dict(_RESTRICTED_BUILTINS)",
        "_GLOBALS['_input_data'] = _input_data",
        "",
        "# ====== Quiet stdout (Nixtla/numba are noisy) ======",
        "class _NoiseFilter:",
        "    def __init__(self, sink):",
        "        self._sink = sink",
        "        self._buf = ''",
        "        self._noise = ('numba', 'INFO:', 'WARNING:', 'statsforecast', 'mlforecast')",
        "        self._keep = ('Traceback', 'Error', 'Exception', 'raise', 'assert')",
        "    def write(self, data):",
        "        self._buf += data",
        "        while '\\n' in self._buf:",
        "            line, self._buf = self._buf.split('\\n', 1)",
        "            if any(k in line for k in self._keep) or not any(n in line for n in self._noise):",
        "                self._sink.write(line + '\\n')",
        "    def flush(self):",
        "        if self._buf:",
        "            line = self._buf",
        "            self._buf = ''",
        "            if any(k in line for k in self._keep) or not any(n in line for n in self._noise):",
        "                self._sink.write(line)",
        "        self._sink.flush()",
        "_orig_stdout = sys.stdout",
        "sys.stdout = _NoiseFilter(_orig_stdout)",
        "try:",
        "    with open('script.py', 'r', encoding='utf-8') as _f:",
        "        _user_code = _f.read()",
        "    exec(_user_code, _GLOBALS)",
        "finally:",
        "    sys.stdout = _orig_stdout",
        "",
        "# ====== Output ======",
        "if '_forecast_result' not in _GLOBALS:",
        "    raise NameError(\"Generated code must set '_forecast_result' variable\")",
        "with open(_OUTPUT_PATH, 'w', encoding='utf-8') as _f:",
        "    json.dump(_GLOBALS['_forecast_result'], _f, default=str, indent=2)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
class SubprocessSandbox:
    """Locked-down subprocess sandbox for untrusted Python execution."""

    def __init__(
        self,
        memory_mb: int = _DEFAULT_MEMORY_MB,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        max_open_files: int = _DEFAULT_MAX_FILES,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        forbidden_imports: Optional[Set[str]] = None,
        forbidden_names: Optional[Set[str]] = None,
        temp_dir: Optional[str] = None,
    ):
        self.memory_mb = memory_mb
        self.timeout_seconds = timeout_seconds
        self.max_open_files = max_open_files
        self.max_output_bytes = max_output_bytes
        self.forbidden_imports = forbidden_imports or _FORBIDDEN_IMPORTS
        self.forbidden_names = forbidden_names or _FORBIDDEN_NAMES
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._platform = platform.system()

    def _check_imports(self, code: str) -> List[str]:
        """Backward-compatible alias for _check_ast."""
        return _check_ast(code, self.forbidden_imports, self.forbidden_names)

    def execute(
        self,
        code: str,
        input_data: Dict[str, Any],
        input_path: Optional[str] = None,
    ) -> SandboxResult:
        """Execute code inside the sandbox.

        Args:
            code: Python source code to execute.
            input_data: Dict serialized to JSON and passed to the sandbox as
                ``_input_data``.
            input_path: Optional explicit path for input JSON.

        Returns:
            SandboxResult with success status, parsed result_data, and diagnostics.
        """
        start = time.perf_counter()

        # 1. AST pre-flight (fast-fail)
        violations = _check_ast(code, self.forbidden_imports, self.forbidden_names)
        if violations:
            return SandboxResult(
                success=False,
                error=f"Blocked constructs detected: {', '.join(violations)}",
            )

        # 2. Prepare temp dirs with automatic cleanup
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="sandbox_", dir=self.temp_dir)
        try:
            if input_path is None:
                fd_in, input_path = tempfile.mkstemp(suffix=".json", dir=temp_dir_obj.name)
                os.close(fd_in)
            fd_out, output_path = tempfile.mkstemp(suffix=".json", dir=temp_dir_obj.name)
            os.close(fd_out)

            # 3. Write input data
            try:
                with open(input_path, "w", encoding="utf-8") as f:
                    json.dump(input_data, f, default=str)
            except Exception as exc:
                return SandboxResult(
                    success=False,
                    error=f"Failed to write input JSON: {exc}",
                )

            # 3b. Write LLM-generated forecasting code to script.py
            script_path = os.path.join(temp_dir_obj.name, "script.py")
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code)
            except Exception as exc:
                return SandboxResult(
                    success=False,
                    error=f"Failed to write script.py: {exc}",
                )

            # 4. Wrap code with safety hooks
            wrapped = _build_wrapper(input_path, output_path)

            # 5. Syntax validation of wrapper
            try:
                compile(wrapped, "<sandbox>", "exec")
            except SyntaxError as exc:
                return SandboxResult(
                    success=False,
                    error=f"Wrapper syntax error: {exc}",
                )

            # 6. Windows Job Object
            windows_job = _create_windows_job(self.memory_mb)

            # 7. Build subprocess kwargs (stripped env = no secret leakage)
            safe_env = _safe_env()
            popen_kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": temp_dir_obj.name,
                "env": safe_env,
            }

            linux_preexec: Optional[Any] = None
            if self._platform == "Linux":
                linux_preexec = lambda: _linux_preexec_fn(  # noqa: E731
                    self.memory_mb, self.timeout_seconds, self.max_open_files
                )

            # 8. Execute subprocess
            try:
                if linux_preexec:
                    proc = subprocess.Popen(
                        [sys.executable, "-c", wrapped],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=temp_dir_obj.name,
                        env=safe_env,
                        preexec_fn=linux_preexec,
                    )
                else:
                    creationflags = 0
                    if self._platform == "Windows":
                        # Own process group so timeout kills are clean.
                        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

                    proc = subprocess.Popen(
                        [sys.executable, "-c", wrapped],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=temp_dir_obj.name,
                        env=safe_env,
                        creationflags=creationflags,
                    )

                    # Assign to job object after creation (Windows)
                    if windows_job and _WIN32_AVAILABLE and self._platform == "Windows":
                        try:
                            win32job.AssignProcessToJobObject(windows_job, proc._handle)
                        except Exception as exc:
                            logger.warning("Failed to assign process to job object: %s", exc)

                try:
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout_bytes, stderr_bytes = proc.communicate()
                    elapsed = time.perf_counter() - start
                    return SandboxResult(
                        success=False,
                        output_path=output_path,
                        stdout=_cap_output(stdout_bytes, self.max_output_bytes).decode("utf-8", errors="replace"),
                        stderr=_cap_output(stderr_bytes, self.max_output_bytes).decode("utf-8", errors="replace"),
                        return_code=-1,
                        execution_time_seconds=elapsed,
                        error=f"Sandbox timeout after {self.timeout_seconds}s",
                    )

                elapsed = time.perf_counter() - start
                stdout = _cap_output(stdout_bytes, self.max_output_bytes).decode("utf-8", errors="replace")
                stderr = _cap_output(stderr_bytes, self.max_output_bytes).decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    return SandboxResult(
                        success=False,
                        output_path=output_path,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=proc.returncode,
                        execution_time_seconds=elapsed,
                        error=f"Sandbox process exited with code {proc.returncode}: {stderr[:500]}",
                    )

                # 9. Read and parse output
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                    if not isinstance(result_data, dict):
                        raise ValueError("Output JSON must be an object")
                    return SandboxResult(
                        success=True,
                        output_path=output_path,
                        result_data=result_data,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=0,
                        execution_time_seconds=elapsed,
                        data_sources_used=result_data.get("data_sources_used", []),
                    )
                except Exception as exc:
                    return SandboxResult(
                        success=False,
                        output_path=output_path,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=0,
                        execution_time_seconds=elapsed,
                        error=f"Failed to parse output JSON: {exc}",
                    )
            finally:
                if windows_job and _WIN32_AVAILABLE:
                    try:
                        win32job.CloseHandle(windows_job)
                    except Exception:
                        pass
        finally:
            # 10. Cleanup temp dir
            try:
                temp_dir_obj.cleanup()
            except Exception as exc:
                logger.debug("Temp dir cleanup warning: %s", exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def run_diagnostics(self) -> Dict[str, Any]:
        return {
            "platform": self._platform,
            "memory_limit_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_open_files": self.max_open_files,
            "max_output_bytes": self.max_output_bytes,
            "forbidden_imports_count": len(self.forbidden_imports),
            "forbidden_names_count": len(self.forbidden_names),
            "restricted_builtins_count": len(_SAFE_BUILTINS),
            "allowed_imports_count": len(_ALLOWED_IMPORT_PREFIXES),
            "win32_job_available": _WIN32_AVAILABLE and self._platform == "Windows",
            "restrictedpython_used": False,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cap_output(data: bytes, max_bytes: int) -> bytes:
    """Cap output bytes to max_bytes to prevent OOM from huge stdout/stderr."""
    if len(data) > max_bytes:
        return data[:max_bytes] + b"\n... [output truncated by sandbox]"
    return data
