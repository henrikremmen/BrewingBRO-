from pathlib import Path
import os
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from brew_utils import *  # noqa: E402,F403


OUT = Path(__file__).resolve().parent / "generated_tables"
OUT.mkdir(exist_ok=True)

barley_cost = barley["cost"].to_numpy(dtype=float)
barley_ebc = barley["EBC"].to_numpy(dtype=float)
barley_q = barley["quality"].to_numpy(dtype=float)

malt_cost = malts["cost"].to_numpy(dtype=float)
malt_ebc = malts["EBC"].to_numpy(dtype=float)
malt_q = malts["quality"].to_numpy(dtype=float)

extract_cost = maltextracts["cost"].to_numpy(dtype=float)
extract_ebc = maltextracts["EBC"].to_numpy(dtype=float)
extract_q = maltextracts["quality"].to_numpy(dtype=float)

hop_cost = hops["cost"].to_numpy(dtype=float)
hop_q = hops["quality"].to_numpy(dtype=float)

yeast_cost = yeasts["cost"].to_numpy(dtype=float)
yeast_q = yeasts["quality"].to_numpy(dtype=float)

F_malt = float(maltprocess.loc[0, "fixedcost"])
v_malt = float(maltprocess.loc[0, "variablecost"])
F_mash = float(mashingprocess.loc[0, "fixedcost"])
v_mash = float(mashingprocess.loc[0, "variablecost"])

nB = len(barley)
nM = len(malts)
nE = len(maltextracts)
nH = len(hops)
nY = len(yeasts)

iB0 = 0
iM0 = iB0 + nB
iE0 = iM0 + nM
iH0 = iE0 + nE
iY0 = iH0 + nH
iZm = iY0 + nY
iZs = iZm + 1
nvars = iZs + 1


def slice_B():
    return slice(iB0, iB0 + nB)


def slice_M():
    return slice(iM0, iM0 + nM)


def slice_E():
    return slice(iE0, iE0 + nE)


def slice_H():
    return slice(iH0, iH0 + nH)


def slice_Y():
    return slice(iY0, iY0 + nY)


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def money(value):
    return fmt(value, 3)


def fmt_ebc(value, color=None):
    if color == "black":
        return fmt(value, 6)
    return fmt(value, 3)


def tex_escape(text):
    return str(text).replace("_", r"\_")


def write_table(filename, body):
    (OUT / filename).write_text(body, encoding="utf-8")


def route_breakdown(route, v_beer):
    m_eq = malt_extract_eq_needed_for_beer(v_beer)
    h_kg = hops_needed_for_beer(v_beer)
    y_kg = yeast_needed_for_beer(v_beer)
    hop_component = h_kg * cheapest_cost(hops)
    yeast_component = y_kg * cheapest_cost(yeasts)

    if route == "Barley":
        barley_kg = m_eq / BARLEY_TO_MALTEXTRACT_EQ
        malt_kg = malt_from_barley(barley_kg)
        parts = {
            "source": barley_kg * cheapest_cost(barley),
            "hop": hop_component,
            "yeast": yeast_component,
            "var_malt": v_malt * barley_kg,
            "var_mash": v_mash * malt_kg,
            "fixed_malt": F_malt,
            "fixed_mash": F_mash,
        }
    elif route == "Malt":
        malt_kg = m_eq / MALT_TO_MALTEXTRACT_EQ
        parts = {
            "source": malt_kg * cheapest_cost(malts),
            "hop": hop_component,
            "yeast": yeast_component,
            "var_malt": 0.0,
            "var_mash": v_mash * malt_kg,
            "fixed_malt": 0.0,
            "fixed_mash": F_mash,
        }
    elif route == "Extract":
        parts = {
            "source": m_eq * cheapest_cost(maltextracts),
            "hop": hop_component,
            "yeast": yeast_component,
            "var_malt": 0.0,
            "var_mash": 0.0,
            "fixed_malt": 0.0,
            "fixed_mash": 0.0,
        }
    else:
        raise ValueError(route)

    parts["total"] = sum(parts.values())
    return parts


def export_task1_breakdown():
    rows = []
    for volume in [50, 5000, 25000]:
        for route in ["Barley", "Malt", "Extract"]:
            p = route_breakdown(route, volume)
            rows.append(
                f"{volume} & {route} & {money(p['source'])} & {money(p['hop'])} & "
                f"{money(p['yeast'])} & {money(p['var_malt'])} & {money(p['var_mash'])} & "
                f"{money(p['fixed_malt'])} & {money(p['fixed_mash'])} & {money(p['total'])} \\\\"
            )

    write_table(
        "task31_cost_breakdown.tex",
        r"""\begin{table}[H]
\renewcommand{\arraystretch}{1.12}
\caption{Task 1 Cost Breakdown for Separate Source Routes}
\label{tab:task31-cost-breakdown}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}rlrrrrrrrr@{}}
\toprule
Volume & Route & Source & Hops & Yeast & Var. malt & Var. mash & Fixed malt & Fixed mash & Total \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def build_cost_model(v_beer, ebc_min, ebc_max, quality=None, z_fix=None):
    m_tot = malt_extract_eq_needed_for_beer(v_beer)
    h_tot = hops_needed_for_beer(v_beer)
    y_tot = yeast_needed_for_beer(v_beer)
    gamma = (SG - 1.0) / 0.0344

    m_big_malt = m_tot / BARLEY_TO_MALTEXTRACT_EQ
    m_big_mash = m_tot / MALT_TO_MALTEXTRACT_EQ

    c = np.zeros(nvars)
    c[slice_B()] = barley_cost + v_malt + v_mash * BARLEY_TO_MALT
    c[slice_M()] = malt_cost + v_mash
    c[slice_E()] = extract_cost
    c[slice_H()] = hop_cost
    c[slice_Y()] = yeast_cost
    c[iZm] = F_malt
    c[iZs] = F_mash

    a_eq = []
    b_eq = []

    row = np.zeros(nvars)
    row[slice_B()] = BARLEY_TO_MALTEXTRACT_EQ
    row[slice_M()] = MALT_TO_MALTEXTRACT_EQ
    row[slice_E()] = 1.0
    a_eq.append(row)
    b_eq.append(m_tot)

    row = np.zeros(nvars)
    row[slice_H()] = 1.0
    a_eq.append(row)
    b_eq.append(h_tot)

    row = np.zeros(nvars)
    row[slice_Y()] = 1.0
    a_eq.append(row)
    b_eq.append(y_tot)

    a_ub = []
    b_ub = []

    if np.isfinite(ebc_max):
        row = np.zeros(nvars)
        row[slice_B()] = gamma * BARLEY_TO_MALTEXTRACT_EQ * barley_ebc
        row[slice_M()] = gamma * MALT_TO_MALTEXTRACT_EQ * malt_ebc
        row[slice_E()] = gamma * extract_ebc
        a_ub.append(row)
        b_ub.append(ebc_max * m_tot)

    row = np.zeros(nvars)
    row[slice_B()] = -gamma * BARLEY_TO_MALTEXTRACT_EQ * barley_ebc
    row[slice_M()] = -gamma * MALT_TO_MALTEXTRACT_EQ * malt_ebc
    row[slice_E()] = -gamma * extract_ebc
    a_ub.append(row)
    b_ub.append(-ebc_min * m_tot)

    if quality is not None:
        q_malt, q_hop, q_yeast = quality

        row = np.zeros(nvars)
        row[slice_B()] = -BARLEY_TO_MALTEXTRACT_EQ * barley_q
        row[slice_M()] = -MALT_TO_MALTEXTRACT_EQ * malt_q
        row[slice_E()] = -extract_q
        a_ub.append(row)
        b_ub.append(-q_malt * m_tot)

        row = np.zeros(nvars)
        row[slice_H()] = -hop_q
        a_ub.append(row)
        b_ub.append(-q_hop * h_tot)

        row = np.zeros(nvars)
        row[slice_Y()] = -yeast_q
        a_ub.append(row)
        b_ub.append(-q_yeast * y_tot)

    row = np.zeros(nvars)
    row[slice_B()] = 1.0
    row[iZm] = -m_big_malt
    a_ub.append(row)
    b_ub.append(0.0)

    row = np.zeros(nvars)
    row[slice_B()] = BARLEY_TO_MALT
    row[slice_M()] = 1.0
    row[iZs] = -m_big_mash
    a_ub.append(row)
    b_ub.append(0.0)

    row = np.zeros(nvars)
    row[iZm] = 1.0
    row[iZs] = -1.0
    a_ub.append(row)
    b_ub.append(0.0)

    bounds = [(0, None)] * nvars
    if z_fix is None:
        bounds[iZm] = (0, 1)
        bounds[iZs] = (0, 1)
    else:
        bounds[iZm] = (z_fix[0], z_fix[0])
        bounds[iZs] = (z_fix[1], z_fix[1])

    return {
        "c": c,
        "A_eq": np.array(a_eq, dtype=float),
        "b_eq": np.array(b_eq, dtype=float),
        "A_ub": np.array(a_ub, dtype=float),
        "b_ub": np.array(b_ub, dtype=float),
        "bounds": bounds,
        "m_tot": m_tot,
        "H_tot": h_tot,
        "Y_tot": y_tot,
    }


def solve_model(model):
    return linprog(
        c=model["c"],
        A_ub=model["A_ub"],
        b_ub=model["b_ub"],
        A_eq=model["A_eq"],
        b_eq=model["b_eq"],
        bounds=model["bounds"],
        method="highs",
    )


def solve_bruteforce(v_beer, ebc_min, ebc_max, quality=None):
    best = None
    all_results = []
    for z_fix in [(0, 0), (0, 1), (1, 1)]:
        model = build_cost_model(v_beer, ebc_min, ebc_max, quality=quality, z_fix=z_fix)
        res = solve_model(model)
        info = {
            "z_fix": z_fix,
            "success": res.success,
            "cost": res.fun if res.success else np.nan,
            "model": model,
            "result": res,
        }
        all_results.append(info)
        if res.success and (best is None or res.fun < best["cost"]):
            best = info
    return best, all_results


def solution_summary(model, res):
    x = res.x
    b = x[slice_B()]
    m = x[slice_M()]
    e = x[slice_E()]
    h = x[slice_H()]
    y = x[slice_Y()]
    m_tot = model["m_tot"]
    h_tot = model["H_tot"]
    y_tot = model["Y_tot"]
    gamma = (SG - 1.0) / 0.0344

    ebc = gamma * (
        BARLEY_TO_MALTEXTRACT_EQ * np.sum(barley_ebc * b)
        + MALT_TO_MALTEXTRACT_EQ * np.sum(malt_ebc * m)
        + np.sum(extract_ebc * e)
    ) / m_tot

    q_malt = (
        BARLEY_TO_MALTEXTRACT_EQ * np.sum(barley_q * b)
        + MALT_TO_MALTEXTRACT_EQ * np.sum(malt_q * m)
        + np.sum(extract_q * e)
    ) / m_tot
    q_hop = np.sum(hop_q * h) / h_tot
    q_yeast = np.sum(yeast_q * y) / y_tot

    return {
        "cost": res.fun,
        "EBC": ebc,
        "Q_malt": q_malt,
        "Q_hop": q_hop,
        "Q_yeast": q_yeast,
        "z_malt": int(round(x[iZm])),
        "z_mash": int(round(x[iZs])),
    }


def nonzero_ingredients(x, tol=1e-7):
    groups = [
        ("Barley", list(barley["name"]), x[slice_B()]),
        ("Malt", list(malts["name"]), x[slice_M()]),
        ("Malt extract", list(maltextracts["name"]), x[slice_E()]),
        ("Hops", list(hops["name"]), x[slice_H()]),
        ("Yeast", list(yeasts["name"]), x[slice_Y()]),
    ]
    rows = []
    for group, names, values in groups:
        for name, value in zip(names, values):
            if abs(value) > tol:
                rows.append((group, name, float(value)))
    return rows


def ingredient_mix_text(x):
    rows = nonzero_ingredients(x)
    return "; ".join(f"{tex_escape(name)} {fmt(value)} kg" for _, name, value in rows)


def export_task34_solutions():
    summary_rows = []
    ingredient_rows = []
    for volume in [50, 25000]:
        best, _ = solve_bruteforce(volume, 20, 29, quality=None)
        summary = solution_summary(best["model"], best["result"])
        summary_rows.append(
            f"{volume} & $({summary['z_malt']},{summary['z_mash']})$ & "
            f"{money(summary['cost'])} & {fmt(summary['EBC'])} & "
            f"{fmt(summary['Q_malt'])} & {fmt(summary['Q_hop'])} & {fmt(summary['Q_yeast'])} \\\\"
        )
        for group, name, value in nonzero_ingredients(best["result"].x):
            ingredient_rows.append(
                f"{volume} & {group} & \\texttt{{{tex_escape(name)}}} & {fmt(value)} \\\\"
            )

    write_table(
        "task34_solution_summary.tex",
        r"""\begin{table}[H]
\renewcommand{\arraystretch}{1.15}
\caption{Task 4 MILP Solution Summary for Additional Volumes}
\label{tab:task34-solution-summary}
\centering
\small
\begin{tabular}{@{}rcrrrrr@{}}
\toprule
Volume (L) & Process & Cost & EBC & $Q_S$ & $Q_H$ & $Q_Y$ \\
\midrule
"""
        + "\n".join(summary_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )
    write_table(
        "task34_nonzero_ingredients.tex",
        r"""\begin{table}[H]
\renewcommand{\arraystretch}{1.12}
\caption{Task 4 Non-Zero Ingredients for 50 L and 25000 L}
\label{tab:task34-nonzero-ingredients}
\centering
\small
\begin{tabular}{@{}rllr@{}}
\toprule
Volume (L) & Group & Variable & Amount (kg) \\
\midrule
"""
        + "\n".join(ingredient_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def export_task36_solutions():
    summary_rows = []
    ingredient_rows = []
    for color, (ebc_min, ebc_max) in EBC_RANGES.items():
        best, _ = solve_bruteforce(5000, ebc_min, ebc_max, quality=(3, 3, 3))
        if best is None:
            continue
        summary = solution_summary(best["model"], best["result"])
        summary_rows.append(
            f"{color} & $({summary['z_malt']},{summary['z_mash']})$ & "
            f"{money(summary['cost'])} & {fmt_ebc(summary['EBC'], color)} & "
            f"{fmt(summary['Q_malt'])} & {fmt(summary['Q_hop'])} & {fmt(summary['Q_yeast'])} \\\\"
        )
        ingredient_rows.append(
            f"{color} & {ingredient_mix_text(best['result'].x)} \\\\"
        )

    write_table(
        "task36_solution_summary.tex",
        r"""\begin{table}[H]
\renewcommand{\arraystretch}{1.15}
\caption{Task 6 Solution Summary for Feasible Beer Colours}
\label{tab:task36-solution-summary}
\centering
\small
\begin{tabular}{@{}lcrrrrr@{}}
\toprule
Colour & Process & Cost & EBC & $Q_S$ & $Q_H$ & $Q_Y$ \\
\midrule
"""
        + "\n".join(summary_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )
    write_table(
        "task36_nonzero_ingredients.tex",
        r"""\begin{table}[H]
\renewcommand{\arraystretch}{1.12}
\caption{Task 6 Non-Zero Ingredient Mixes for Feasible Colours}
\label{tab:task36-nonzero-ingredients}
\centering
\scriptsize
\begin{tabular}{@{}lp{0.78\textwidth}@{}}
\toprule
Colour & Non-zero ingredients \\
\midrule
"""
        + "\n".join(ingredient_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


if __name__ == "__main__":
    export_task1_breakdown()
    export_task34_solutions()
    export_task36_solutions()
