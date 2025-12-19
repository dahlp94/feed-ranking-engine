import json
from pathlib import Path
import yaml

from src.frec.data.simulator import simulate_data, split_sessions_by_time, MINUTES_PER_DAY
from src.frec.viz.diagnostics import run_diagnostics

def main():
    cfg = yaml.safe_load(open("configs/dev.yaml", "r"))
    data = simulate_data(cfg)

    train_days = int(cfg["split"]["train_days"])
    val_days = int(cfg["split"]["val_days"])
    train_end = train_days * MINUTES_PER_DAY
    val_end = (train_days + val_days) * MINUTES_PER_DAY

    train, val, test = split_sessions_by_time(data.sessions, train_end, val_end)

    # For diagnostics, focus on TRAIN so we understand the data we learn from.
    # Use "now" as end of train for popularity window plots.
    now_ts = train_end

    figs_dir = Path("outputs/figures")
    summary = run_diagnostics(
        data=data,
        sessions=train,
        figs_dir=figs_dir,
        now_ts=now_ts,
    )

    summary_path = Path("outputs/diagnostics_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary.__dict__, indent=2))

    print("Saved figures to:", figs_dir.resolve())
    print("Saved summary to:", summary_path.resolve())
    print(json.dumps(summary.__dict__, indent=2))

if __name__ == "__main__":
    main()
