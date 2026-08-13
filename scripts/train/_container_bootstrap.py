#!/usr/bin/env python
"""Java + pyserini bootstrap shared by the container-side dense build entrypoints.

The eval containers ship without pyserini or a JDK, so any job that
needs first-stage retrieval installs both at runtime.

Container-side only: this runs inside the training image, installs system
packages, and assumes it may mutate the environment. Nothing here is imported by
`src/presentation_dependence/`.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict | None = None, check: bool = True) -> int:
    """Run a subprocess, echoing the command; raise on failure when ``check``."""
    print(f"[build] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if check and rc != 0:
        raise SystemExit(f"[build][FATAL] command failed (rc={rc}): {' '.join(cmd)}")
    return rc


def _find_libjvm() -> Path | None:
    """Locate libjvm.so anywhere under the common JDK roots (jnius needs it)."""
    roots = [
        "/usr/lib/jvm",
        os.environ.get("CONDA_PREFIX", "/opt/conda"),
        "/opt/conda",
        "/usr/local",
    ]
    for root in roots:
        for hit in glob.glob(f"{root}/**/libjvm.so", recursive=True):
            return Path(hit)
    return None


def _install_java_and_deps(pip_extra: list[str]) -> tuple[str, str]:
    """Install Java 21 + the pyserini build deps; return (JAVA_HOME, libjvm_dir).

    jnius (pyserini's JVM bridge) needs ``$JAVA_HOME/lib/server/libjvm.so`` and
    libjvm.so on the loader path. Install via apt (predictable layout) with a
    conda fallback, then locate libjvm.so directly and derive JAVA_HOME from it,
    which works wherever the package landed.
    """
    _run(
        [
            "bash",
            "-lc",
            "apt-get update -y >/dev/null 2>&1; "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-21-jdk-headless "
            "|| DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jdk-headless || true",
        ],
        check=False,
    )
    lib = _find_libjvm()
    if lib is None:
        conda = shutil.which("conda") or shutil.which("mamba")
        if conda:
            _run([conda, "install", "-y", "-c", "conda-forge", "openjdk=21"], check=False)
            lib = _find_libjvm()
    if lib is None:
        raise SystemExit("[build][FATAL] could not install/locate a JDK (libjvm.so not found).")

    libjvm_dir = str(lib.parent)  # .../lib/server
    # Java 9+: JAVA_HOME is three levels up from .../lib/server.
    java_home = str(lib.parent.parent.parent)
    if not Path(java_home, "bin", "java").exists():
        # conda layout: libjvm at $PREFIX/lib/server -> JAVA_HOME = $PREFIX.
        java_home = str(lib.parent.parent)

    # Pin pyserini to 1.4.0. Newer builds added an `encode._openai` module that
    # eagerly instantiates an OpenAI client at import and crashes without
    # OPENAI_API_KEY (pulled in via the impact/dense search import chain).
    pkgs = [
        "pyserini==1.4.0",
        "datasets>=2.18,<5",
        "faiss-cpu>=1.8.0",
        "onnxruntime",
        "pytrec-eval>=0.5",
        "pyyaml>=6.0.1",
        *pip_extra,
    ]
    _run([sys.executable, "-m", "pip", "install", "--no-input", *pkgs], check=True)
    # Some PyTorch base images ship a torch/torchvision pair whose ABI breaks
    # transformers' lazy import (`operator torchvision::nms does not exist`),
    # which pyserini's ONNX-encoder path pulls in. The dense build needs neither
    # (encoding is onnxruntime, search is Lucene/jnius), so remove them to keep
    # transformers importable.
    _run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"], check=False)
    print(f"[build] JAVA_HOME={java_home}  libjvm_dir={libjvm_dir}", flush=True)
    return java_home, libjvm_dir


def _code_root() -> Path:
    """Locate the shipped code bundle, identified by the dense setup driver.

    A job runner uploads ``scripts/train/`` as the source dir and copies
    ``scripts/`` alongside it, so the marker is resolved relative to the
    entrypoint before falling back to the conventional code path and cwd.
    """
    marker = Path("scripts") / "data" / "setup_msmarco_self_distill_dense.py"
    here = Path(__file__).resolve()
    # The final candidate retains compatibility with legacy container bundles
    for base in (here.parent, here.parent.parent, here.parent.parent.parent, Path("/opt/ml/code")):
        if (base / marker).exists():
            return base
    return Path.cwd()


def _build_env(java_home: str, libjvm_dir: str) -> dict:
    """Return the environment the dense-build driver expects."""
    env = dict(os.environ)
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{java_home}/bin:" + env.get("PATH", "")
    # jnius dlopen's libjvm.so via the loader path; keep its dir on it.
    env["LD_LIBRARY_PATH"] = f"{libjvm_dir}:" + env.get("LD_LIBRARY_PATH", "")
    env["PY"] = sys.executable  # driver uses $PY (no `uv` in the container)
    env["PARADIGMS"] = "dense"
    env["MSMARCO_DENSE"] = "hnsw"  # no Lucene-flat BGE for MS MARCO; HNSW ef=1000 ~ exact
    env["ROBUST_DOWNLOAD"] = "1"
    env["RUN_DIAGNOSTICS"] = "1"
    env["PURGE_AFTER"] = "1"  # reclaim scratch disk between surfaces
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Some pyserini builds eagerly init an OpenAI client at import, which needs
    # the variable to be non-empty. The client is never called (ONNX BGE
    # encoder), and the value deliberately does not look like a key so secret
    # scanners do not flag it.
    env.setdefault("OPENAI_API_KEY", "unused-no-openai-calls-in-this-job")
    return env
