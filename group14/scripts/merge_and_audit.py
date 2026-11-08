"""
merge_and_audit.py - merge the v2 CSVs into one master file and report whether they are comparable.

  python merge_and_audit.py runs_savvas_v2.csv runs_teammate_v2.csv --out runs_master_v0.csv

Rows are matched by column name. Nothing is deleted: decide about flagged runs as a team.
"""
import argparse

import numpy as np
import pandas as pd

MUST_MATCH = ["max_steps", "num_epoch", "learning_rate_schedule", "time_horizon", "summary_freq", "hidden_units",
              "num_layers", "time_scale", "no_graphics", "num_envs", "num_areas", "env_file", "mlagents_version"]
FACTORS = ["learning_rate", "batch_size", "buffer_size"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--out", default="runs_master_v0.csv")
    args = parser.parse_args()

    data = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True, sort=False)
    data["config_id"] = data[FACTORS].astype(str).agg("|".join, axis=1)
    data["block"] = data["source"].astype(str) + "|" + data["cpu_model"].astype(str)
    blockers = 0

    print("Runs: %d" % len(data))
    print(data.groupby("source").size().to_string())

    duplicates = data[data["run_id"].duplicated(keep=False)]["run_id"].unique()
    if len(duplicates):
        blockers += 1
        print("\nBLOCKER duplicate run_id: %s" % list(duplicates))

    print("\nSETTINGS THAT MUST MATCH")
    for column in MUST_MATCH:
        values = data[column].fillna("MISSING").astype(str).value_counts()
        if len(values) > 1:
            blockers += 1
            print("BLOCKER %s differs:" % column)
            print(pd.crosstab(data["source"], data[column].fillna("MISSING").astype(str)).to_string())
        else:
            print("ok  %s = %s" % (column, values.index[0]))

    print("\nDATA QUALITY")
    print("incomplete: %d" % (~data["completed"].astype(bool)).sum())
    print("seed not recorded: %d (usable, but not reproducible)" % (~data["seed_recorded"].astype(bool)).sum())
    print("not converged: %d of %d" % ((~data["converged"].astype(bool)).sum(), len(data)))

    print("\nDESIGN")
    print("distinct configs: %d" % data["config_id"].nunique())
    print("hardware blocks: %d" % data["block"].nunique())
    print("runs per config: min %d, median %.1f, max %d" % (
        data.groupby("config_id").size().min(), data.groupby("config_id").size().median(),
        data.groupby("config_id").size().max()))
    print("\nbatch x buffer:")
    print(pd.crosstab(data["batch_size"], data["buffer_size"]).to_string())
    print("\nlearning rate x batch:")
    print(pd.crosstab(data["learning_rate"], data["batch_size"]).to_string())
    print("\nseed x source:")
    print(pd.crosstab(data["random_seed"], data["source"]).to_string())

    print("\nPROJECT PLAN COVERAGE: distinct recorded seeds per LR x batch, buffer 2048 only (target 3)")
    plan = data[(data["buffer_size"] == 2048) & data["seed_recorded"].astype(bool)
                & data["learning_rate"].isin([0.0001, 0.0003, 0.001]) & data["batch_size"].isin([64, 128, 256])]
    table = plan.pivot_table(index="learning_rate", columns="batch_size", values="random_seed",
                             aggfunc="nunique", fill_value=0)
    table = table.reindex(index=[0.0001, 0.0003, 0.001], columns=[64, 128, 256], fill_value=0)
    print(table.to_string())

    shared = data.groupby("config_id")["source"].nunique()
    print("\nconfigs shared by more than one source: %d" % (shared > 1).sum())

    groups = data.groupby(["block", "config_id"])["final_reward"]
    variances = groups.var().dropna()
    if len(variances):
        counts = groups.count()
        counts = counts[counts > 1]
        dof = int((counts - 1).sum())
        pooled = float(np.sqrt(((counts - 1) * variances).sum() / dof))
        print("\nseed noise in final_reward: pooled SD %.3f (%d degrees of freedom)" % (pooled, dof))

    data.to_csv(args.out, index=False)
    print("\nWrote %s" % args.out)
    print("VERDICT: %s" % ("STOP, fix the blockers above before modelling" if blockers
                           else "settings match, data can be pooled"))
    raise SystemExit(1 if blockers else 0)


if __name__ == "__main__":
    main()