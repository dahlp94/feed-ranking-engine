import json
from pathlib import Path
import yaml

from src.frec.data.simulator import simulate_data, split_sessions_by_time, MINUTES_PER_DAY
from src.frec.retrieval.covisit import build_covisitation_index, evaluate_candidate_recall


def main():
    cfg = yaml.safe_load(open("configs/dev.yaml", "r"))
    data = simulate_data(cfg)

    # time split
    train_days = int(cfg["split"]["train_days"])
    val_days = int(cfg["split"]["val_days"])
    train_end = train_days * MINUTES_PER_DAY
    val_end = (train_days + val_days) * MINUTES_PER_DAY

    train, val, test = split_sessions_by_time(data.sessions, train_end, val_end)

    # sort sessions by time (important for leakage-free history)
    train_sorted = sorted(train, key=lambda s: s.ts)
    val_sorted = sorted(val, key=lambda s: s.ts)
    test_sorted = sorted(test, key=lambda s: s.ts)

    ccfg = cfg["covisitation"]
    idx = build_covisitation_index(
        sessions=train_sorted,
        n_items=len(data.posts),
        max_neighbors_per_item=int(ccfg["max_neighbors_per_item"]),
        use_clicked_only=bool(ccfg["use_clicked_only"]),
        max_items_per_session=int(ccfg["max_items_per_session"]),
        session_pair_window=int(ccfg["session_pair_window"]),
        downsample_impressions=bool(ccfg["downsample_impressions"]),
        seed=int(cfg["seed"]),
    )


    # Build history from TRAIN only (then evaluate VAL and TEST sequentially)
    history_by_user = {}
    for s in train_sorted:
        if s.clicked_post_ids:
            history_by_user.setdefault(s.user_id, [])
            history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[: int(ccfg["max_history_items"]) * 5]

    k_list = cfg["eval"]["k_list"]

    report_val = evaluate_candidate_recall(
        sessions_sorted=val_sorted,
        index=idx,
        history_by_user={u: h.copy() for u, h in history_by_user.items()},
        topk_per_history_item=int(ccfg["topk_per_history_item"]),
        max_history_items=int(ccfg["max_history_items"]),
        retrieval_k=int(ccfg["retrieval_k"]),
        k_list=k_list,
    )

    report_test = evaluate_candidate_recall(
        sessions_sorted=test_sorted,
        index=idx,
        history_by_user={u: h.copy() for u, h in history_by_user.items()},
        topk_per_history_item=int(ccfg["topk_per_history_item"]),
        max_history_items=int(ccfg["max_history_items"]),
        retrieval_k=int(ccfg["retrieval_k"]),
        k_list=k_list,
    )

    out = {
        "counts": {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_items": len(data.posts),
        },
        "covisitation_config": ccfg,
        "candidate_recall_val": report_val,
        "candidate_recall_test": report_test,
    }

    out_path = Path("outputs/covisitation_candidate_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
