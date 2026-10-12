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
    log_dir: str = "/data/user_data/jgibson2/asr_sprint/results/rq1/logs",
    extra_sbatch_lines: list[str] | None = None,
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
        Directory for Slurm stdout/stderr log files.
    extra_sbatch_lines:
        Additional ``#SBATCH`` directives (without the ``#SBATCH`` prefix).
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={log_dir}/{job_name}_%j.log",
        "#SBATCH --signal=B:USR1@60",
    ]

    if extra_sbatch_lines:
        for line in extra_sbatch_lines:
            lines.append(f"#SBATCH {line}")

    lines += [
        "",
        f"source {venv_path}/bin/activate",
        f"export PYTHONPATH={project_root}:$PYTHONPATH",
        "export HF_HOME=/data/user_data/jgibson2/asr_sprint/hf_cache",
        "export TRANSFORMERS_CACHE=/data/user_data/jgibson2/asr_sprint/model_cache",
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
