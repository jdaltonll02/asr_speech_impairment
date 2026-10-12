"""
Slurm job script utilities for the CDLI Rwandan NSS ASR sprint.

Provides helpers to write ``.slurm`` job files and optionally submit them via
``sbatch``.  All jobs use the ``preempt`` partition.

Usage::

    from src.utils.slurm import write_slurm_script, submit_job
    write_slurm_script(
        path="jobs/my_job.slurm",
        job_name="my_job",
        command="python -m src.rq1.baseline --resume",
        partition="preempt",
        gres="gpu:1",
        mem="32G",
        time="04:00:00",
        venv_path="/data/user_data/jgibson2/asr_sprint/venv",
        project_root="/home/jgibson2/projects/asr-sprint",
    )
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_slurm_script(
    path: str,
    job_name: str,
    command: str,
    partition: str,
    gres: str,
    mem: str,
    time: str,
    venv_path: str,
    project_root: str,
    log_dir: str | None = None,
    extra_sbatch_lines: list[str] | None = None,
    hf_home: str | None = None,
    transformers_cache: str | None = None,
) -> None:
    """Write a ``.slurm`` batch script to *path*.

    Parameters
    ----------
    path:
        Destination file path (should end in ``.slurm``).
    job_name:
        Slurm job name (``--job-name``).
    command:
        Shell command(s) to execute (can be multi-line).
    partition:
        Slurm partition name.  Must be ``'preempt'`` for this project.
    gres:
        Generic resource string, e.g. ``'gpu:1'``.
    mem:
        Memory request, e.g. ``'32G'``.
    time:
        Wall-clock time limit, e.g. ``'04:00:00'``.
    venv_path:
        Absolute path to the Python virtual environment to activate.
    project_root:
        Absolute path to the project root (added to ``PYTHONPATH``).
    log_dir:
        Directory for Slurm stdout/stderr log files.  When ``None`` the
        caller must ensure the ``--output`` directive is supplied via
        *extra_sbatch_lines*, or a sensible default derived from *cfg* should
        be passed by the caller.
    extra_sbatch_lines:
        Additional ``#SBATCH`` directives (without the ``#SBATCH`` prefix).
    hf_home:
        Value for the ``HF_HOME`` environment variable.  Derived from the
        config ``data_root`` by the caller; no default applied here so that
        no path is ever hardcoded in this function.
    transformers_cache:
        Value for the ``TRANSFORMERS_CACHE`` environment variable.  Same
        rationale as *hf_home*.
    """
    output_directive = (
        f"{log_dir}/{job_name}_%j.log" if log_dir else f"{job_name}_%j.log"
    )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={output_directive}",
        "#SBATCH --signal=B:USR1@60",
    ]

    if extra_sbatch_lines:
        for line in extra_sbatch_lines:
            lines.append(f"#SBATCH {line}")

    lines += [
        "",
        f"source {venv_path}/bin/activate",
        f"export PYTHONPATH={project_root}:$PYTHONPATH",
    ]

    if hf_home is not None:
        lines.append(f"export HF_HOME={hf_home}")
    if transformers_cache is not None:
        lines.append(f"export TRANSFORMERS_CACHE={transformers_cache}")

    lines += [
        "",
        command,
    ]

    script_content = "\n".join(lines) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script_content, encoding="utf-8")


def submit_job(path: str) -> str:
    """Submit a Slurm job script and return the job ID.

    Parameters
    ----------
    path:
        Path to the ``.slurm`` script.

    Returns
    -------
    str
        Slurm job ID string (e.g. ``'12345'``).

    Raises
    ------
    RuntimeError
        If ``sbatch`` returns a non-zero exit code.
    """
    result = subprocess.run(
        ["sbatch", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {path}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    # Expected output: "Submitted batch job 12345"
    parts = result.stdout.strip().split()
    job_id = parts[-1] if parts else ""
    return job_id
