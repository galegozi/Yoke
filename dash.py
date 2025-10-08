import os
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress_bar import ProgressBar
from rich.text import Text
from rich.live import Live

# === CONFIGURATION ===
IMAGING_DIR = Path("applications/evaluation")
PROD_BASE_DIR = Path("applications/harnesses/chicoma_lsc_loderunner-ch-subsampling/runs")
IMAGING_TARGET = 6
DEFAULT_IMAGING_DURATION = timedelta(minutes=40)
PROD_FALLBACK_MINUTES = {
    "gpu": 440,
    "gpu_debug": 54,
    "debug": 54
}

console = Console()

# === HELPERS ===

def parse_time_used(time_str):
    try:
        parts = list(map(int, time_str.strip().split(":")))
        if len(parts) == 2:
            return timedelta(minutes=parts[0], seconds=parts[1])
        elif len(parts) == 3:
            return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
    except:
        pass
    return timedelta()

def parse_start_time(start_str):
    try:
        if start_str == "N/A":
            return None
        return datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
    except:
        return None

def format_eta_time(dt):
    if not dt or not isinstance(dt, datetime):
        return "Unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def render_timeline_bar(start: datetime, end: datetime, width: int = 20) -> Text:
    now = datetime.now()
    total = (end - start).total_seconds()
    elapsed = (now - start).total_seconds()
    pct = min(max(elapsed / total, 0), 1.0) if total > 0 else 0
    filled = int(pct * width)
    bar = "[" + "█" * filled + "-" * (width - filled) + "]"
    return Text(bar)

# === JOB LOG PARSING ===

def imaging_progress(job_name):
    out_path = IMAGING_DIR / f"{job_name}.out"
    if not out_path.exists():
        return 0
    with open(out_path, 'r', errors='ignore') as f:
        return sum(1 for line in f if "In lsc_loderunner_anime main" in line)

def prod_progress_and_avg(job_name):
    match = re.match(r"exp(\d{3})_ep(\d{4})", job_name)
    if not match:
        return "0/10", "-", None
    exp_id, epoch_start = match.groups()
    epoch_start = int(epoch_start)
    epoch_end = epoch_start + 9
    log_path = PROD_BASE_DIR / f"study_{exp_id}" / f"study{exp_id}_epoch{epoch_start:04d}.out"
    if not log_path.exists():
        return "0/10", "-", None

    with open(log_path, 'r', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    epoch_times = []
    current_epoch = None

    for i in range(len(lines)):
        line = lines[i]
        match_epoch = re.search(r"Completed epoch (\d+)", line)
        if match_epoch:
            current_epoch = int(match_epoch.group(1))
            for j in range(i + 1, min(i + 5, len(lines))):
                if "Epoch time (minutes):" in lines[j]:
                    try:
                        minutes = float(lines[j].split(":")[-1].strip())
                        epoch_times.append((current_epoch, minutes))
                        break
                    except:
                        continue

    if not epoch_times:
        return "0/10", "-", None

    completed_epochs = [e for e, _ in epoch_times if epoch_start <= e <= epoch_end]
    done = len(completed_epochs)
    last_time = next((t for e, t in reversed(epoch_times) if e in completed_epochs), None)
    normal = [t for e, t in epoch_times if e % 5 != 0 and epoch_start <= e <= epoch_end]
    valid = [t for e, t in epoch_times if e % 5 == 0 and epoch_start <= e <= epoch_end]

    avg_norm = sum(normal) / len(normal) if normal else 5
    avg_val = sum(valid) / len(valid) if valid else 2

    remaining = 0
    for e in range(epoch_start + done, epoch_end + 1):
        remaining += avg_val if e % 5 == 0 else avg_norm

    ep_time_str = f"{last_time:.1f} min" if last_time else "-"
    return f"{done}/10", ep_time_str, timedelta(minutes=remaining)

# === SLURM PARSING ===

def parse_slurm_jobs():
    cmd = ["squeue", "--Format=jobid,name,state,timeused,starttime,partition", "--noheader", "--me"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        jobid, name, state, timeused, starttime, partition = parts
        state = state.upper()
        if state not in ("R", "PD", "RUNNING", "PENDING"):
            continue
        job_type = "imaging" if name.startswith("imgen") else "prod" if name.startswith("exp") else None
        if job_type:
            jobs.append({
                "id": jobid,
                "name": name,
                "type": job_type,
                "state": state,
                "timeused": timeused,
                "starttime": starttime,
                "partition": partition.lower()
            })
    # ensure that no prod jobs are missing.
    # ids = [
    #     102, 200, 201, 205, 210, 225, 250, 300
    # ]
    # filtered = [j for j in jobs if j['type'] == 'prod']
    # for i in ids:
    #     # see if there is a job with that id
    #     for j in filtered:
    #         if j['name'].startswith(f'exp{i}'):
    #             break
    #     else:
    #         print(i)
    return jobs

def get_eta_datetime(job, chain_offset=None):
    now = datetime.now()
    state = job["state"].upper()
    elapsed = parse_time_used(job["timeused"])
    start_dt = parse_start_time(job["starttime"])
    partition = job["partition"]

    if job["type"] == "imaging":
        steps = imaging_progress(job["name"])
        if state == "RUNNING":
            if steps > 0:
                avg_step = elapsed / steps
                return now + avg_step * (IMAGING_TARGET - steps)
            return now + (DEFAULT_IMAGING_DURATION - elapsed)
        else:
            base = chain_offset or start_dt or now
            return base + DEFAULT_IMAGING_DURATION

    elif job["type"] == "prod":
        _, _, remaining = prod_progress_and_avg(job["name"])
        fallback = timedelta(minutes=PROD_FALLBACK_MINUTES.get(partition, 54))
        if state == "RUNNING":
            raw_eta = remaining if remaining else fallback - elapsed
            return max(now, now + raw_eta)
        else:
            base = chain_offset or start_dt or now
            return base + (remaining if remaining else fallback)

    return now + timedelta(minutes=60)

# === DASHBOARD ===

def build_dashboard_table():
    jobs = parse_slurm_jobs()

    gpu_debug_pending = sorted(
        [j for j in jobs if j["partition"] == "gpu_debug" and j["state"] == "PENDING"],
        key=lambda j: parse_start_time(j["starttime"]) or datetime.now()
    )

    chain_start = datetime.now()
    gpu_debug_eta_map = {}
    for job in gpu_debug_pending:
        if job["type"] == "imaging":
            steps = imaging_progress(job["name"])
            duration = DEFAULT_IMAGING_DURATION if steps == 0 else (DEFAULT_IMAGING_DURATION / IMAGING_TARGET) * (IMAGING_TARGET - steps)
        elif job["type"] == "prod":
            _, _, remaining = prod_progress_and_avg(job["name"])
            duration = remaining or timedelta(minutes=PROD_FALLBACK_MINUTES["gpu_debug"])
        gpu_debug_eta_map[job["name"]] = chain_start
        chain_start += duration

    job_entries = []
    for job in jobs:
        chain_offset = gpu_debug_eta_map.get(job["name"])
        eta_dt = get_eta_datetime(job, chain_offset)
        job_entries.append((eta_dt, job, chain_offset))
    job_entries.sort(key=lambda tup: tup[0])

    table = Table(title="🎛️ Active SLURM Jobs")
    table.add_column("Job")
    table.add_column("Type")
    table.add_column("State")
    table.add_column("Partition")
    table.add_column("Progress")
    table.add_column("Epoch Time")
    table.add_column("ETA")
    table.add_column("Progress Bar")
    table.add_column("Timeline")

    for eta_dt, job, chain_offset in job_entries:
        eta_str = format_eta_time(eta_dt)
        state = job["state"]

        if job["type"] == "imaging":
            steps = imaging_progress(job["name"])
            bar = ProgressBar(total=IMAGING_TARGET, completed=steps, width=20)
            start_time = datetime.now() - parse_time_used(job["timeused"]) if state == "RUNNING" else (parse_start_time(job["starttime"]) or datetime.now())
            timeline = render_timeline_bar(start_time, eta_dt)
            table.add_row(job["name"], "Imaging", state, job["partition"], f"{steps}/6", "-", eta_str, bar, timeline)

        elif job["type"] == "prod":
            progress, ep_time, _ = prod_progress_and_avg(job["name"])
            done = int(progress.split("/")[0]) if "/" in progress else 0
            bar = ProgressBar(total=10, completed=done, width=20)
            start_time = datetime.now() - parse_time_used(job["timeused"]) if state == "RUNNING" else (parse_start_time(job["starttime"]) or datetime.now())
            timeline = render_timeline_bar(start_time, eta_dt)
            table.add_row(job["name"], "Prod", state, job["partition"], progress, ep_time, eta_str, bar, timeline)

    return table

def display_dashboard_loop():
    with Live(build_dashboard_table(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                time.sleep(10)
                live.update(build_dashboard_table())
        except KeyboardInterrupt:
            console.print("[red]Dashboard stopped.")

if __name__ == "__main__":
    display_dashboard_loop()