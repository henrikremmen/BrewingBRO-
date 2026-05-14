import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc



SECTION_COLUMNS = {
    "barley": ["name", "EBC", "cost", "quality"],
    "hops": ["name", "cost", "quality"],
    "maltextracts": ["name", "EBC", "cost", "quality"],
    "maltprocess": ["name", "fixedcost", "variablecost"],
    "malts": ["name", "EBC", "cost", "quality"],
    "mashingprocess": ["name", "fixedcost", "variablecost"],
    "yeasts": ["name", "cost", "quality"],
}

def load_tables(path="dataset_EN.txt"):
    tables = {section: [] for section in SECTION_COLUMNS}
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                label = line[1:].strip()
                if label in SECTION_COLUMNS:
                    current_section = label
                continue

            if current_section is None:
                continue

            parts = line.split("\t")
            expected_len = len(SECTION_COLUMNS[current_section])

            if len(parts) == expected_len:
                tables[current_section].append(parts)

    dataframes = {}
    for section, rows in tables.items():
        df = pd.DataFrame(rows, columns=SECTION_COLUMNS[section])

        for col in df.columns:
            if col != "name":
                df[col] = pd.to_numeric(df[col])

        dataframes[section] = df

    return dataframes


data = load_tables("dataset_EN.txt")

barley = data["barley"]
hops = data["hops"]
maltextracts = data["maltextracts"]
maltprocess = data["maltprocess"]
malts = data["malts"]
mashingprocess = data["mashingprocess"]
yeasts = data["yeasts"]

# display(barley.head())
# display(hops.head())
# display(maltextracts.head())
# display(maltprocess.head())
# display(malts.head())
# display(mashingprocess.head())
# display(yeasts.head())

#test
# barley01_ebc = barley.iloc[0]["EBC"]
# print(barley01_ebc)


# Wort / beer relations
SG = 1.050
EVAPORATION_FRACTION = 0.05
BEER_YIELD_FROM_WORT = 1.0 - EVAPORATION_FRACTION   # 0.95

# m_tot / V_wort = 2.9 * (SG - 1)
# This gives required malt-extract-equivalent mass per liter wort
MALTEXTRACT_EQ_PER_L_WORT = 2.9 * (SG - 1)   # kg / L wort

# Conversion factors
BARLEY_TO_MALT = 0.75                  # 1 kg barley -> 0.75 kg malt
MALT_TO_MALTEXTRACT_EQ = 0.8           # 1 kg malt -> 0.8 kg malt-extract equivalent
BARLEY_TO_MALTEXTRACT_EQ = BARLEY_TO_MALT * MALT_TO_MALTEXTRACT_EQ   # 0.6

# Additives
HOPS_G_PER_L_WORT = 1.3
HOPS_KG_PER_L_WORT = HOPS_G_PER_L_WORT / 1000.0

# 75 g yeast per 100 L cooled liquid
YEAST_G_PER_100L_BEER = 75.0
YEAST_KG_PER_L_BEER = YEAST_G_PER_100L_BEER / 1000.0 / 100.0   # 0.00075 kg/L

# Beer color classes
EBC_RANGES = {
    "light blond": (6, 8),
    "blond": (9, 12),
    "gold": (13, 19),
    "amber": (20, 29),
    "copper": (30, 45),
    "brown": (46, 75),
    "dark brown": (76, 120),
    "black": (120, np.inf),
}



def wort_from_beer_volume(v_beer):
    """
    Convert final beer volume to pre-boil wort volume.
    V_beer = 0.95 * V_wort
    """
    return v_beer / BEER_YIELD_FROM_WORT


def beer_from_wort_volume(v_wort):
    """
    Convert pre-boil wort volume to final beer volume.
    """
    return BEER_YIELD_FROM_WORT * v_wort


def malt_extract_eq_needed_for_wort(v_wort):
    """
    Required malt-extract-equivalent mass for a given wort volume.
    """
    return MALTEXTRACT_EQ_PER_L_WORT * v_wort


def malt_extract_eq_needed_for_beer(v_beer):
    """
    Required malt-extract-equivalent mass for a given final beer volume.
    """
    v_wort = wort_from_beer_volume(v_beer)
    return malt_extract_eq_needed_for_wort(v_wort)


def hops_needed_for_wort(v_wort):
    """
    Hops needed in kg for a given wort volume.
    """
    return HOPS_KG_PER_L_WORT * v_wort


def hops_needed_for_beer(v_beer):
    """
    Hops needed in kg for a given final beer volume.
    Uses wort volume before evaporation.
    """
    v_wort = wort_from_beer_volume(v_beer)
    return hops_needed_for_wort(v_wort)


def yeast_needed_for_beer(v_beer):
    """
    Yeast needed in kg for a given final beer volume.
    Yeast is added to the cooled liquid after boiling.
    In this simplified setup, that is the final beer volume.
    """
    return YEAST_KG_PER_L_BEER * v_beer


def malt_from_barley(barley_kg):
    """
    Convert barley mass to malt mass.
    """
    return BARLEY_TO_MALT * barley_kg


def malt_extract_eq_from_malt(malt_kg):
    """
    Convert malt mass to malt-extract-equivalent mass.
    """
    return MALT_TO_MALTEXTRACT_EQ * malt_kg


def malt_extract_eq_from_barley(barley_kg):
    """
    Convert barley mass directly to malt-extract-equivalent mass.
    """
    return BARLEY_TO_MALTEXTRACT_EQ * barley_kg


def wort_from_barley(barley_kg):
    """
    Compute wort volume obtainable from a given barley mass.
    """
    m_eq = malt_extract_eq_from_barley(barley_kg)
    return m_eq / MALTEXTRACT_EQ_PER_L_WORT


def beer_from_barley(barley_kg):
    """
    Compute final beer volume obtainable from a given barley mass.
    """
    return beer_from_wort_volume(wort_from_barley(barley_kg))


def equivalent_barley_needed_for_beer(v_beer):
    """
    Barley mass needed if all fermentable material comes from barley.
    """
    m_eq = malt_extract_eq_needed_for_beer(v_beer)
    return m_eq / BARLEY_TO_MALTEXTRACT_EQ


def equivalent_malt_needed_for_beer(v_beer):
    """
    Malt mass needed if all fermentable material comes from malt.
    """
    m_eq = malt_extract_eq_needed_for_beer(v_beer)
    return m_eq / MALT_TO_MALTEXTRACT_EQ


def equivalent_maltextract_needed_for_beer(v_beer):
    """
    Malt extract mass needed if all fermentable material comes from malt extract.
    """
    return malt_extract_eq_needed_for_beer(v_beer)


def weighted_average(values, weights):
    """
    Weighted average with zero-protection.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_weight = weights.sum()
    if total_weight <= 0:
        return 0.0

    return np.sum(values * weights) / total_weight


def ebc_of_malt_source_mix(ebc_values, malt_extract_eq_masses):
    """
    EBC of the malt-source mixture using formula (2) from the assignment.
    Inputs must be in malt-extract-equivalent masses.
    """
    m_tot = np.sum(malt_extract_eq_masses)
    if m_tot <= 0:
        return 0.0

    factor = (SG - 1.0) / 0.0344
    weighted_ebc = weighted_average(ebc_values, malt_extract_eq_masses)
    return factor * weighted_ebc


def quality_of_mix(quality_values, masses):
    """
    Weighted mean quality for any ingredient group.
    For malt sources, use malt-extract-equivalent masses.
    For hops and yeasts, use actual masses.
    """
    return weighted_average(quality_values, masses)


def cheapest_row(df, cost_col="cost"):
    """
    Return the row with minimum cost.
    """
    return df.loc[df[cost_col].idxmin()]


def cheapest_cost(df, cost_col="cost"):
    """
    Return the minimum cost value in a dataframe.
    """
    return float(df[cost_col].min())

