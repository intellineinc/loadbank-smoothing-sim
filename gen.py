import csv
import time
import json
import numpy as np
import random

cycle = 0
rpo = 0
_FIELDNAMES = ["cycle", "tpo", "rpo"]
_STAGES = [0.1, 1, 2, 5, 10]
_MAX_QTY = [100, 1, 2, 1, 6]  # max number of each stage type available
_VAR = 1 + np.random.uniform(-0.1, 0.1, len(_STAGES))

_MAX_RATE = 5  # rate of increase when linearly interpolating
_UNDERSHOOT = 5  # underserve power when ramping up (ensure b/w 0 and 10)

# DIFFERENT SMOOTHING ALGOS
def naive_approach(tpo: float, prev_rpo: float) -> np.array:
    qtys = [0] * len(_STAGES)
    diff = tpo

    for idx in range(len(_STAGES) - 1, -1, -1):
        qty = min(_MAX_QTY[idx], diff // _STAGES[idx])
        diff -= qty * _STAGES[idx]
        qtys[idx] = qty

    return np.array(qtys)


def naive_linear(tpo: float, prev_rpo: float) -> np.array:
    qtys = [0] * len(_STAGES)
    diff = min(tpo, prev_rpo + _MAX_RATE if tpo >= prev_rpo else prev_rpo - _MAX_RATE)

    for idx in range(len(_STAGES) - 1, -1, -1):
        qty = min(_MAX_QTY[idx], diff // _STAGES[idx])
        diff -= qty * _STAGES[idx]
        qtys[idx] = qty

    return np.array(qtys)


def naive_proportional(tpo: float, prev_rpo: float) -> np.array:
    qtys = [0] * len(_STAGES)
    diff = min(tpo, prev_rpo + (tpo - prev_rpo) * 0.1)

    for idx in range(len(_STAGES) - 1, -1, -1):
        qty = min(_MAX_QTY[idx], diff // _STAGES[idx])
        diff -= qty * _STAGES[idx]
        qtys[idx] = qty

    return np.array(qtys)


def optimal_linear(tpo: float, prev_rpo: float) -> np.array:
    # calculate target by considering undershoot and maximum change/cycle
    qtys = [0] * len(_STAGES)
    target = min(
        tpo - _UNDERSHOOT,
        prev_rpo + _MAX_RATE if tpo >= prev_rpo else prev_rpo - _MAX_RATE,
    )

    # loop through stages to determine config
    diff = target
    for idx in range(len(_STAGES) - 1, -1, -1):
        qty = min(_MAX_QTY[idx], diff // _STAGES[idx])
        diff -= qty * _STAGES[idx]
        qtys[idx] = qty

    # check calculated rpo -> if within range, make up diff with cvs
    qtys = np.array(qtys)
    calc_rpo = np.array(_STAGES) @ qtys
    cvs_gap = (tpo - calc_rpo) // _STAGES[0]
    if (cvs_gap > 0 and cvs_gap <= (_MAX_QTY[0] - qtys[0])) or (
        cvs_gap < 0 and abs(cvs_gap) <= qtys[0]
    ):
        qtys[0] += cvs_gap

    return np.array(qtys)


def choose_stage_qty(tpo: float, prev_rpo: float) -> np.array:
    return naive_proportional(tpo, prev_rpo)


# MAIN SCRIPT
if __name__ == "__main__":
    with open("data.csv", "w") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        csv_writer.writeheader()

    while True:
        # check tpo from json
        with open("tpo.json", "r") as f:
            data = json.load(f)
            tpo = data["tpo"]

        # get previous row data
        with open("data.csv", "r") as f1:
            prev_cycle, prev_tpo, prev_rpo = f1.readlines()[-1].split(",")
            try:
                prev_rpo = float(prev_rpo)
            except Exception as e:
                prev_rpo = 0

        # decision making to achieve rpo
        with open("data.csv", "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
            info = {"cycle": cycle, "tpo": tpo, "rpo": rpo}
            csv_writer.writerow(info)

            cycle += 1
            stage_qty = choose_stage_qty(tpo, prev_rpo)
            rpo = (np.array(_STAGES)) @ stage_qty
            print(stage_qty)

        time.sleep(0.5)
