#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import logging
import math


def log_ratio(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan

    return math.log10(a / b)


def read_midas_output(
    file_path,
    selected_weeks,
    selected_samples,
    min_abundance=None,
    min_reads=None,
    infer_sample_name=True,
):
    # file_path = midas_output
    # selected_weeks = sample_weeks
    logging.info("Reading Midas output ...")
    df = pd.read_csv(
        file_path,
        header=0,
        usecols=[
            "name",
            "sample_id",
            "count_reads",
            "relative_abundance",
            "Week",
            "Mouse",
            "Challenge",
        ],
    ).dropna(subset=["Mouse"])

    if min_abundance is not None:
        df = (
            df.query(
                f"(Week in @selected_weeks) and (relative_abundance >= {min_abundance})"
            )
            .reset_index(drop=True)
            .copy()
        )

    if min_reads is not None:
        df = (
            df.query(f"(Week in @selected_weeks) and (count_reads >= {min_reads})")
            .reset_index(drop=True)
            .copy()
        )

    if infer_sample_name:
        df["sample_name"] = df.apply(lambda x: f"{x.Week}{x.Mouse}", axis=1)
    else:
        df["sample_name"] = df["sample_id"]

    df = df.query("sample_name in @selected_samples").copy()

    df["unique_name"] = df.apply(lambda x: f"{x.sample_name}---{x['name']}", axis=1)
    # By default Midas relative abundances are between 0-1, make them between 0-100
    # to match NM
    df["midas_relative_abundance"] = df["relative_abundance"] * 100
    df = df[
        [
            "unique_name",
            "name",
            "sample_name",
            "count_reads",
            "midas_relative_abundance",
        ]
    ]
    df.columns = [
        "unique_name",
        "midas_bucket",
        "sample_name",
        "count_reads",
        "midas_relative_abundance",
    ]
    logging.info(
        f"{df['midas_bucket'].nunique()} buckets across {df['sample_name'].nunique()} samples found."
    )
    logging.info(f"{df['unique_name'].nunique()} bucket-sample pairs found.")
    return df


def read_strain_midas_contributions(
    contributions_file, min_contrib=0, trigger_contrib_check=20
):
    # contributions_file = strain_midas_contribution_file
    logging.info("Reading Strain --> Midas contributions output ...")
    select_cols = {
        "sample_id": "Organism",
        "name": "midas_bucket",
        "abundance": "strain_contribution",
    }
    # new_col_names = ["Organism", "midas_bucket", "strain_contribution"]
    init_df = pd.read_csv(
        contributions_file, header=0, usecols=select_cols.keys()
    ).rename(columns=select_cols)

    upper = 100 - trigger_contrib_check
    lower = trigger_contrib_check
    offmapping_df = init_df.query("@lower <= strain_contribution <= @upper")
    num_offmapped_strains, _ = offmapping_df.shape
    if num_offmapped_strains > 0:
        logging.warning(
            f"\nFound {num_offmapped_strains} instances of significant (> {trigger_contrib_check}%) strain offmapping to midas buckets in contributions data."
            f"This is usually caused due to inaccurate/erroneous genome assemblies. Please make sure that this is not the case, before relying on the output of this analysis."
        )
        logging.warning(f"\n{offmapping_df}")

    df = init_df.query(f"strain_contribution >= {min_contrib}")
    logging.info(
        f"{df['Organism'].nunique()} actual strains map to {df['midas_bucket'].nunique()} Midas buckets"
    )
    # acceptable_buckets = set(df["midas_bucket"])
    primary_buckets, secondary_buckets = find_primary_bucket(df)
    shared_buckets = find_buckets_linked_by_strains(init_df, primary_buckets)

    return (df, shared_buckets)


def find_primary_bucket(df):
    # Primary buckets:
    # - group by strain,
    # - descending order,
    # - pick top bucket
    # Secondary buckets: Midas buckets to which more than one of our strains map,
    # but they are not the primary catchment for any of our strains
    df = pd.read_csv(strain_midas_contribution_file)

    # Group by strains (sample_id) and keep the top contributor bucket (name) by abundance
    primary_df = (
        df.groupby(["sample_id"])
        .apply(lambda x: x.sort_values(["abundance"], ascending=False))
        .reset_index(drop=True)
        .groupby(["sample_id"])
        .head(1)
    )

    primary_buckets = set(primary_df["name"])
    secondary_df = (
        df.query("name not in @primary_buckets")
        .sort_values(["name"])
        .reset_index(drop=True)
    )
    secondary_buckets = sorted(
        secondary_df["name"].unique()
    )  # as list for alphabetical printing
    logging.info(
        f"Found {len(secondary_buckets)} midas buckets that are not primary catchments for any strain:"
    )
    _ = [logging.info(f"\t{b}") for b in secondary_buckets]

    return primary_buckets, set(secondary_buckets)
    # return primary_buckets


def find_buckets_linked_by_strains(df, acceptable_buckets):
    linked_buckets = dict()
    for row in df.itertuples(index=False):
        if row.Organism in linked_buckets:
            linked_buckets[row.Organism].add(row.midas_bucket)
        else:
            linked_buckets[row.Organism] = {row.midas_bucket}

    shared_buckets = dict()
    for group in linked_buckets.values():
        if len(group) == 1:
            continue
        for bucket in group:
            if bucket in acceptable_buckets:
                continue
            if bucket in shared_buckets:
                shared_buckets[bucket] |= group
            else:
                shared_buckets[bucket] = set(group)

    num_shared_strains = len(shared_buckets.keys())
    logging.info(
        f"{num_shared_strains} instances found where a single strain contributed to multiple midas buckets."
    )
    return shared_buckets


def read_nm_output(file_path, read_stats, selected_samples, min_abundance=1e-6):
    logging.info("Reading NinjaMap output ...")
    read_stats_df = pd.read_csv(read_stats, header=0)

    read_stats_df["adjustment"] = (
        read_stats_df["Fragments_Aligned"] / read_stats_df["Fragments_After_Trim"]
    )
    keep_cols = ["sample_id", "adjustment"]
    read_stats_df = read_stats_df[keep_cols]

    df = (
        pd.read_csv(file_path, header=0)
        .query("(sample_id in @selected_samples)")
        .reset_index(drop=True)
        .merge(read_stats_df, how="left", on="sample_id")
        .assign(Norm_Read_Fraction=lambda x: x.Read_Fraction * x.adjustment)
        .query(f"Norm_Read_Fraction >= {min_abundance}")
    )[
        [
            "sample_id",
            "Strain_Name",
            "Norm_Read_Fraction",
            "Percent_Coverage",
            "Coverage_Depth",
        ]
    ]

    # Update column names to match outputs from other tools.
    df.columns = [
        "sample_name",
        "Organism",
        "NM_Norm_Read_Fraction",
        "Percent_Coverage",
        "Coverage_Depth",
    ]
    logging.info(
        f"{df['Organism'].nunique()} strains across {len(selected_samples)} samples found."
    )
    return df


def nm2midas(nm_df, strain_midas_df):
    logging.info(
        "Transforming strain level abundances into midas bucket level abundances ..."
    )
    # join_type = "outer" # default
    join_type = "left"  # testing
    nm2midas = nm_df.merge(right=strain_midas_df, how=join_type, on="Organism").dropna(
        subset=["sample_name"]
    )

    bucket_strain_weights = (
        strain_midas_df[["midas_bucket", "strain_contribution"]]
        .groupby(["midas_bucket"])
        .agg({"strain_contribution": np.nansum,})
        .reset_index()
        .rename(columns={"strain_contribution": "total_strain_contribution"})
        .dropna(subset=["total_strain_contribution"])
    )

    strain_weighted_abundance = nm2midas.merge(
        right=bucket_strain_weights, how="left", on=["midas_bucket"]
    ).assign(
        proportional_contribution=lambda x: x.strain_contribution
        / x.total_strain_contribution,
        proportional_rel_abund=lambda x: x.NM_Norm_Read_Fraction
        * x.proportional_contribution,
    )[
        ["sample_name", "midas_bucket", "Organism", "proportional_rel_abund"]
    ]

    nm2midas_imputed = (
        strain_weighted_abundance.groupby(["sample_name", "midas_bucket"])
        .agg({"proportional_rel_abund": np.nansum})
        .reset_index()
        .rename(columns={"proportional_rel_abund": "imputed_NM_rel_abund"})
    )

    nm2midas_imputed["unique_name"] = nm2midas_imputed.apply(
        lambda x: f"{x.sample_name}---{x.midas_bucket}", axis=1
    )

    nm2midas_imputed = nm2midas_imputed[["unique_name", "imputed_NM_rel_abund"]]
    # nm2midas_imputed.sort_values(
    #     "imputed_NM_rel_abund", ascending=False
    # ).reset_index(drop=True)
    logging.info(
        f"Imputed NinjaMap strains --> Midas buckets: {nm2midas_imputed['unique_name'].nunique()} bucket-sample pairs"
    )
    return nm2midas_imputed, strain_weighted_abundance


def fix_extreme_pos_invaders(
    current_pred, extreme_pos, min_extreme_pos=2, alt_prediction="Unclear"
):
    if current_pred.lower() == "invader":
        if extreme_pos < min_extreme_pos:
            return alt_prediction

    return current_pred


def read_ic_output(file_path, sample_names, min_extreme_pos=2):
    logging.info("Reading invaderCheck output ...")
    df = pd.read_csv(file_path, header=0).dropna(subset=["Prediction"])
    df["sample_name"] = df["Sample"].apply(lambda x: x.split("_")[2])
    df_selected = df.query("sample_name in @sample_names")

    # TODO @sunit: This is a temporary fix; the data needs to be run through the comparison
    # step again for a more final version
    # df_selected["Prediction"] = df_selected.apply(
    #     lambda row: fix_extreme_pos_invaders(row.Prediction, row.Extreme_Positions),
    #     axis=1,
    # )

    logging.info(
        f"Category counts (strains):\n{df_selected['Prediction'].value_counts()}"
    )
    logging.info(
        f"{df_selected['Organism'].nunique()} strains across {df_selected['sample_name'].nunique()} samples found."
    )
    return df_selected


def ic2midas(ic_df, strain_midas_df):
    logging.info(
        "Aggregating strain level invader predictions into midas bucket level predictions ..."
    )
    # join_type = "outer" # default
    join_type = "left"  # testing
    ic_pred = (
        ic_df.merge(right=strain_midas_df, how=join_type, on="Organism",)
        .reset_index(drop=True)[["sample_name", "midas_bucket", "Prediction"]]
        .drop_duplicates()
        .groupby(["sample_name", "midas_bucket"])
        .agg(unique_IC_pred=("Prediction", "unique"))
        .reset_index()
    )

    # ic_pred.sort_values("IC_Final_Prediction")
    ic_pred["unique_name"] = ic_pred.apply(
        lambda x: f"{x.sample_name}---{x.midas_bucket}", axis=1
    )
    ic_pred["unique_IC_pred"] = ic_pred["unique_IC_pred"].apply(lambda x: frozenset(x))
    ic_pred["unique_IC_pred_len"] = ic_pred["unique_IC_pred"].apply(lambda x: len(x))
    # ic_pred.sort_values("unique_IC_pred_len", ascending=False)
    ic_pred.drop(["sample_name", "midas_bucket"], axis=1, inplace=True)
    logging.info(
        f"Category counts (buckets):\n{ic_pred['unique_IC_pred'].value_counts()}"
    )
    logging.info(
        f"Imputed IC strains --> Midas buckets: {ic_pred['unique_name'].nunique()} bucket-sample pairs"
    )
    return ic_pred[["unique_name", "unique_IC_pred", "unique_IC_pred_len"]]


def aggregate_across_tools(midas_df, nm_df, ic_df):
    logging.info("Aggregating data from all tools in midas buckets terms ...")
    df = midas_df.merge(
        right=nm_df,
        how="outer",
        on=["unique_name"],
        suffixes=["_midas", "_NM"],
        validate="one_to_one",
    ).merge(right=ic_df, how="outer", on=["unique_name"], suffixes=["_midas", "_IC"],)

    df["midas_NM_ratio_log10"] = df.apply(
        lambda x: log_ratio(x.midas_relative_abundance, x.imputed_NM_rel_abund), axis=1
    )

    df = df[
        [
            "unique_name",
            "sample_name",
            "midas_bucket",
            "count_reads",
            "midas_relative_abundance",
            "imputed_NM_rel_abund",
            "midas_NM_ratio_log10",
            "unique_IC_pred",
            "unique_IC_pred_len",
        ]
    ]
    # remove any row that does not have midas relative abundance, since it cannot be used.
    before_midas_trimming = df["unique_name"].nunique()
    logging.info(
        f"Aggregated data across all tools contains {df['unique_name'].nunique()} bucket-sample pairs."
    )
    df.dropna(subset=["midas_relative_abundance"], inplace=True)
    after_midas_trimming = df["unique_name"].nunique()
    logging.info(
        f"Removed {before_midas_trimming - after_midas_trimming} midas-sample combinations that did not have a midas relative abundance"
    )
    logging.info(
        f"Aggregated data across all tools now contains {df['unique_name'].nunique()} bucket-sample pairs."
    )

    lower, upper = get_thresholds(df)
    df["unique_IC_pred"] = df.apply(lambda row: other_ic_pred_categories(row), axis=1)
    df["tool_agreement"] = df.apply(
        lambda row: check_tool_agreement(row, lower, upper), axis=1
    )
    logging.info(
        "Removing cases where a Midas bucket was not found. Since we can't use that data."
    )
    df.dropna(subset=["tool_agreement"], inplace=True)

    df["midas_is_greater"] = df["midas_NM_ratio_log10"].apply(lambda x: x > upper)
    logging.info(f"Category counts (buckets):\n{df['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Aggregated data finally contains {df['unique_name'].nunique()} bucket-sample pairs from {df['midas_bucket'].nunique()} buckets and {df['sample_name'].nunique()} samples."
    )

    return df


def check_tool_agreement(row, lower, upper):
    value = row.midas_NM_ratio_log10
    nm_value = row.imputed_NM_rel_abund
    midas_value = row.midas_relative_abundance

    if np.isnan(nm_value) and np.isnan(midas_value):
        # If neither tool detected a bucket/strain (but IC did)
        return None
    elif (not np.isnan(nm_value)) and np.isnan(midas_value):
        # If Midas did not detect a bucket, but NM did
        return None
    elif np.isnan(nm_value) and (not np.isnan(midas_value)):
        # If NM did not detect a bucket, but Midas did
        return "In_Midas_Only"
    elif lower < value < upper:
        # If Midas and NM found a bucket and both reported relative abundances within tolerable range (2 * SD)
        return "Agree"
    elif (value <= lower) or (upper <= value):
        # If Midas and NM found a bucket and both reported relative abundances outside of tolerable range (2 * SD)
        return "Disagree"
    else:
        # Any other case that hasn't been accounted for.
        return "Error"


def other_ic_pred_categories(row):
    if type(row.unique_IC_pred) == frozenset:
        return row.unique_IC_pred

    if (
        (not np.isnan(row.midas_relative_abundance))
        and (np.isnan(row.imputed_NM_rel_abund))
        and (np.isnan(row.unique_IC_pred))
    ):
        return frozenset({"In_Midas_Only"})
    elif (
        (np.isnan(row.midas_relative_abundance))
        and (not np.isnan(row.imputed_NM_rel_abund))
        and (np.isnan(row.unique_IC_pred))
    ):
        return frozenset({"In_NM_Only"})
    elif np.isnan(row.unique_IC_pred):
        return frozenset({"No_IC_Pred"})
    else:
        return frozenset({"ERROR"})


def get_NM_Midas_agree_df(df, lower, upper):
    return df.query(f"{lower} < midas_NM_ratio_log10 < {upper}").copy()


def get_NM_Midas_disagree_df(df, lower, upper):
    return df.query(
        f"(midas_NM_ratio_log10 <= {lower})  or ({upper} <= midas_NM_ratio_log10)"
    ).copy()


def get_thresholds(df):
    assert (
        "midas_NM_ratio_log10" in df.columns
    ), "This is not the dataframe you're looking for ..."
    dist_mean = df["midas_NM_ratio_log10"].mean(skipna=True)
    dist_sd = df["midas_NM_ratio_log10"].std(skipna=True)
    kc_threshold = dist_sd * 2
    kc_lower = dist_mean - kc_threshold
    kc_upper = dist_mean + kc_threshold
    logging.info("Midas/NM Log10 ratio:")
    logging.info(f"\tMean:{dist_mean}")
    logging.info(f"\tSD:{dist_sd}")
    logging.info(f"\tAcceptable range:{kc_lower} to {kc_upper}")
    return (kc_lower, kc_upper)


def identify_inputs(all_cases_df):
    # NM and Midas agree and IC says Input
    logging.info("Searching for Inputs ...")
    df = all_cases_df.copy()
    df["Final_Prediction"] = df.apply(
        lambda x: "Input"
        if (
            (x.unique_IC_pred_len == 1)
            and ("Input" in x.unique_IC_pred)
            and (x.tool_agreement == "Agree")
        )
        else None,
        axis=1,
    )
    df_selected = df.dropna(subset=["Final_Prediction"]).reset_index(drop=True)
    logging.info(f"Category counts:\n{df_selected['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Final prediction counts:\n{df_selected['Final_Prediction'].value_counts()}"
    )
    logging.info(f"Final prediction buckets:\n{df_selected['midas_bucket'].nunique()}")
    return df_selected


def identify_invaders(all_cases_df):
    logging.info("Searching for Invaders ...")
    df = all_cases_df.copy()
    df["Final_Prediction"] = df.apply(
        lambda x: "Invader"
        # NM and Midas agree and IC says Invader
        if ((x.unique_IC_pred_len == 1) and ("Invader" in x.unique_IC_pred))
        # Only Midas can see it
        or (x.tool_agreement == "In_Midas_Only")
        # NM and Midas disagree and Midas is greater
        or (
            (x.unique_IC_pred_len == 1)
            and (x.tool_agreement == "Disagree")
            and x.midas_is_greater
        )
        else None,
        axis=1,
    )
    df_selected = df.dropna(subset=["Final_Prediction"]).reset_index(drop=True)
    logging.info(f"Category counts:\n{df_selected['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Final prediction counts:\n{df_selected['Final_Prediction'].value_counts()}"
    )
    logging.info(f"Final prediction buckets:\n{df_selected['midas_bucket'].nunique()}")
    return df_selected


def identify_mixed(all_cases_df, already_identified):
    logging.info("Searching for Mixed buckets ...")
    df = all_cases_df.copy()
    logging.info(
        f"Remove {len(already_identified)} unique names that have already been categorized"
    )
    df["Final_Prediction"] = df.query("unique_name not in @already_identified").apply(
        lambda x: "Mixed"
        # Midas bucket has contributions from multiple type of strains
        # if (x.unique_IC_pred_len == 2 and ("Unclear" not in x.unique_IC_pred) )
        if (x.unique_IC_pred == {"Input", "Invader"}) else None,
        axis=1,
    )
    df_selected = df.dropna(subset=["Final_Prediction"]).reset_index(drop=True)
    logging.info(f"Category counts:\n{df_selected['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Final prediction counts:\n{df_selected['Final_Prediction'].value_counts()}"
    )
    logging.info(f"Final prediction buckets:\n{df_selected['midas_bucket'].nunique()}")
    return df_selected


def identify_unclear(all_cases_df, already_identified):
    logging.info("Searching for Unclear predictions ...")
    df = all_cases_df.copy()
    logging.info(
        f"Remove {len(already_identified)} unique names that have already been categorized"
    )
    df["Final_Prediction"] = df.query("unique_name not in @already_identified").apply(
        lambda x: "Unclear"
        if (not np.isnan(x.unique_IC_pred_len)) or ("No_IC_Pred" in x.unique_IC_pred)
        else None,
        axis=1,
    )
    # df["Final_Prediction"] = "Unclear"
    df_selected = df.dropna(subset=["Final_Prediction"]).reset_index(drop=True)
    logging.info(f"Category counts:\n{df_selected['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Final prediction counts:\n{df_selected['Final_Prediction'].value_counts()}"
    )
    logging.info(f"Final prediction buckets:\n{df_selected['midas_bucket'].nunique()}")
    return df_selected


def plot_relative_abundance_comparison(df, file_prefix, color="Final_Prediction"):
    import seaborn as sns

    sns.set()
    sns.set_style("whitegrid")
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (24, 24)

    sns.scatterplot(
        data=df,
        y="imputed_NM_rel_abund",
        x="midas_relative_abundance",
        hue=color,
        palette="colorblind",
    ).set(xscale="log", yscale="log")
    # ylim=(1e-5, 1e2), xlim=(1e-5, 1e2)
    plt.plot([1e-4, 1e2], [1e-4, 1e2], linewidth=2, color="red", linestyle="dashed")
    plt.savefig(
        f"{file_prefix}.png", bbox_inches="tight",
    )
    plt.close()


def plot_bucket_relative_abundance(
    df, sample_order, file_prefix, color="Final_Prediction"
):
    import seaborn as sns

    sns.set()
    sns.set_style("whitegrid")
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (24, 12)

    # mouse_order = [f"W8M{i}" for i in range(1, 18)]
    sns.stripplot(
        data=df,
        y="midas_relative_abundance",
        x="sample_name",
        hue=color,
        palette="colorblind",
        order=sample_order,
        dodge=True,
    ).set(
        yscale="log", title="", ylim=(1e-4, 1e2),
    )
    plt.savefig(
        f"{file_prefix}.png", bbox_inches="tight",
    )
    plt.close()


def add_strain_context_back(
    df, strain_weighted_abundance, invader_check_df, ninjamap_df
):
    # add_strain_context_back(mixed_df, strain_weighted_bucket_nm_abundance, ic_mice_df, nm_df)
    data = (
        df
        # .query(f"midas_bucket == '{bucket}'")
        .merge(
            right=strain_weighted_abundance,  # strain_weighted_bucket_nm_abundance,
            on=["sample_name", "midas_bucket"],
            how="left",
        )
        .merge(
            right=invader_check_df[
                [
                    "sample_name",
                    "Organism",
                    "Prediction",
                    "Extreme_Positions",
                    "missed_2_EP",
                ]
            ],
            on=["sample_name", "Organism"],
            how="left",
        )
        .merge(right=ninjamap_df, on=["sample_name", "Organism"], how="left",)[
            [
                "unique_name",
                "sample_name",
                "midas_bucket",
                "Organism",
                "count_reads",
                "midas_relative_abundance",
                "imputed_NM_rel_abund",
                "proportional_rel_abund",
                "NM_Norm_Read_Fraction",
                "Prediction",
                "missed_2_EP",
                "Final_Prediction",
                "Extreme_Positions",
                "Percent_Coverage",
                "Coverage_Depth",
            ]
        ]
        .rename(
            columns={
                "Prediction": "Org_initial_prediction",
                "Final_Prediction": "Bucket_final_prediction",
                "missed_2_EP": "Org_prob_missed_2_EP",
                "count_reads": "Bucket_midas_num_reads_assigned",
                "midas_relative_abundance": "Bucket_midas_relative_abundance",
                "imputed_NM_rel_abund": "Bucket_imputed_NM_rel_abund",
                "proportional_rel_abund": "Org_proportional_rel_abund",
                "NM_Norm_Read_Fraction": "Org_NM_Norm_Read_Fraction",
                "Extreme_Positions": "Org_Extreme_Positions",
                "Percent_Coverage": "Org_Percent_Coverage",
                "Coverage_Depth": "Org_Coverage_Depth",
            },
        )
    )
    return data


def remove_mismapped_midas_buckets(
    df, linked_buckets, control_samples, remaining_samples
):
    logging.info(
        f"Searching for mismapped buckets in {len(control_samples)} control samples"
    )
    # sample_order = control_samples + remaining_samples
    contaminant_buckets = set(
        df.query(
            "(sample_name in @control_samples) and (tool_agreement == 'In_Midas_Only')"
        )["midas_bucket"].unique()
    )
    logging.info(
        f"{len(contaminant_buckets)} potential mismapped buckets found in the control samples."
    )
    logging.info(
        f"Searching for these buckets in {len(remaining_samples)} remaining samples"
    )
    # Remove buckets that are only present in control samples
    present_outside_saline = set(
        df.query(
            "(midas_bucket in @contaminant_buckets) and (sample_name in @remaining_samples)"
        )["midas_bucket"].unique()
    )

    remove_buckets = set()
    if len(present_outside_saline) > 0:
        # contaminants that are not present outside saline
        remove_buckets = contaminant_buckets.difference(present_outside_saline)
        logging.info(
            f"Computing pearson's correlations for {len(present_outside_saline)} buckets with other related major buckets:"
        )
        for bucket in present_outside_saline:
            if is_correlated_with_another(
                df, linked_buckets, bucket, remaining_samples
            ):
                remove_buckets |= {bucket}

        removed_buckets = "\n\t".join([i for i in remove_buckets])
        logging.info(
            f"Removing the following {len(remove_buckets)} off-mapped buckets from analysis as they are highly correlated with other major buckets.\n\t"
            f"{removed_buckets}"
        )
        clean_df = df.query("midas_bucket not in @remove_buckets").copy()
    else:
        logging.info("No mismapping found across weeks")
        clean_df = df

    logging.info(f"Category counts:\n{clean_df['unique_IC_pred'].value_counts()}")
    logging.info(
        f"Cleaned data finally contains {clean_df['unique_name'].nunique()} bucket-sample pairs from {clean_df['midas_bucket'].nunique()} buckets and {clean_df['sample_name'].nunique()} samples."
    )
    return clean_df, remove_buckets


def is_correlated_with_another(
    df, linked_buckets, bucket, sample_order, min_correlation=0.75
):
    base = extract_bucket_sample_vector(bucket, df, sample_order)
    correlations = list()
    if bucket not in linked_buckets:
        return False
    for b in linked_buckets[bucket]:
        # for all the values in this bucket
        if b in linked_buckets:
            # if the value is also a key in this dict.
            continue
        query = extract_bucket_sample_vector(b, df, sample_order)
        corr = base.corr(query, min_periods=4)
        logging.info(f"{bucket} vs {b} = {corr}")
        correlations.append(corr > min_correlation)
        # correlations.append(base.corr(query, min_periods=4))

    return any(correlations)


def extract_bucket_sample_vector(bucket, df, sample_order):
    # return a pandas series object for midas bucket, ordered by samples
    this_bucket = df.query(f"midas_bucket == '{bucket}'").copy()
    this_bucket.set_index("sample_name", inplace=True)
    return this_bucket["midas_relative_abundance"].reindex(sample_order)


def check_prediction_consistency(df):
    # Logic in R
    # invader_check_df %>%
    #   select(Mouse,Week,midas_bucket,Bucket_final_prediction) %>%
    #   unique() %>%
    #   group_by(Mouse,midas_bucket) %>%
    #   pivot_wider(names_from = Week, values_from=Bucket_final_prediction)
    ####

    # df = df[["midas_bucket", "sample_name", "Final_Prediction"]]
    df = all_categorized_df[["midas_bucket", "sample_name", "Final_Prediction"]].copy()

    df["Mouse"] = df["sample_name"].apply(
        lambda x: int(x.replace("M", " ").split(" ")[1])
    )
    df["Week"] = df["sample_name"].apply(lambda x: x.replace("M", " ").split(" ")[0])

    # df.set_index(['midas_bucket', 'Mouse'], inplace=True)

    logging.info(f"\n{df.sort_values('midas_bucket').head()}")

    # logging.info(f"\n{df.set_index(['midas_bucket', 'Mouse']).stack().head()}")
    # logging.info(
    #     f"\n{df.pivot_table(index=, columns='Week',values='Final_Prediction').head()}"
    # )

    df.drop(columns="sample_name").pivot(
        index=["midas_bucket"], columns=["Week"], values="Final_Prediction"
    )


def parse_sample_metadata(
    sample_metadata, ignore_weeks=["Week0", "Week4"], control_terms=["PBS"]
):
    df = (
        pd.read_csv(sample_metadata, header=0)
        .query("Week != @ignore_weeks")
        .sort_values(["Week", "MouseNum"])
        .reset_index(drop=True)
    )
    sample_weeks = list(df["Week"].unique())
    sample_names = list(df["sample_id"])
    control_samples = list(df.query("Challenge in @control_terms")["sample_id"])
    sample_order = (
        df[["sample_id", "Week", "MouseOrder"]]
        .sort_values(["Week", "MouseOrder"])
        .reset_index(drop=True)
    )

    return sample_weeks, sample_names, control_samples, sample_order


##############################################################################
##############################################################################
##############################################################################

# if __name__ == "__main__":
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
)
date_prefix = 20201231
midas_min_abundance = 0
nm_min_abundance = 0
midas_min_reads = 3

# ic_pred_output = sys.argv[1]
# midas_output = sys.argv[2]
# nm_output = sys.argv[3]
# nm_read_stats = sys.argv[4]
# strain_midas_contribution_file = sys.argv[5]
# sample_week = sys.argv[6]  # use W#, example: W8

## MBFv1 - Test
# dataset_name = "mbfv1"
# ic_pred_output = "s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/output/v3_20201102/majority.03_predictions.csv"
# midas_output = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/Midas_Results/v1/mbfv1_dataframe_minRelAbund_0.csv"
# nm_output = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/NinjaMap/db_SCv1_2/MouseBackFill_V1/analysis/20201105_MouseBackFill_V1.long.csv"
# nm_read_stats = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/NinjaMap/db_SCv1_2/MouseBackFill_V1/analysis/20201105_MouseBackFill_V1.read_stats.csv"
# strain_midas_contribution_file = (
#     "/Users/sunit.jain/Research/Sunit/Midas/art_reads_output.full_dataframe.csv"
# )
# sample_metadata = "/Users/sunit.jain/Research/Alice/Midas_NinjaMap_ImmiCheck_Compare/MBFv1/mbfv1_ic_metadata.csv"
# output_folder = (
#     "/Users/sunit.jain/Research/Alice/Midas_NinjaMap_ImmiCheck_Compare/MBFv1"
# )
# ignored_weeks = ["W0", "W4"]
# challenge_control = ["PBS"]
# infer_sample_names = True

## MBFv2 - Test
dataset_name = "mbfv2"
ic_pred_output = "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2/majority.ic_pred.csv"
midas_output = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/Midas_Results/v2/mbfv2_dataframe_minRelAbund_0.csv"
nm_output = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/NinjaMap/db_SCv2_3/MouseBackFill_V2/analysis/20201231_MouseBackFill_V2.long.csv"
nm_read_stats = "/Users/sunit.jain/Research/Alice/in_vivo/Mouse_Backfill/NinjaMap/db_SCv2_3/MouseBackFill_V2/analysis/20201231_MouseBackFill_V2.read_stats.csv"
strain_midas_contribution_file = (
    "/Users/sunit.jain/Research/Sunit/Midas/SCv2_3.art_reads_output.full_dataframe.csv"
)
sample_metadata = "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2/00_metadata/mbfv2_metadata.csv"
output_folder = (
    "/Users/sunit.jain/Research/Alice/Midas_NinjaMap_ImmiCheck_Compare/MBFv2"
)

ignored_weeks = ["Week0", "Week4"]
challenge_control = ["PBS"]
infer_sample_names = False

output_file = (
    f"{output_folder}/{date_prefix}_{dataset_name.lower()}.input_v_invader.csv"
)

sample_weeks, sample_names, control_samples, sample_order = parse_sample_metadata(
    sample_metadata, ignore_weeks=ignored_weeks, control_terms=challenge_control
)

actual_samples = [sample for sample in sample_names if sample not in control_samples]

# Strain contributions
strain_midas_df, linked_buckets = read_strain_midas_contributions(
    strain_midas_contribution_file, min_contrib=0
)

# Midas output
midas_df = read_midas_output(
    midas_output,
    sample_weeks,
    selected_samples=sample_names,
    min_abundance=midas_min_abundance,
    min_reads=midas_min_reads,
    infer_sample_name=infer_sample_names,
)
# midas_df.sort_values("count_reads").head()

# NinjaMap output
## Normalize by num reads after trim (as opposed to num reads aligned)
nm_df = read_nm_output(nm_output, nm_read_stats, sample_names, nm_min_abundance)

# Transform NinjaMap strain level relative abundance values into Midas bucket
# level relative abundances
nm2midas_df, strain_weighted_bucket_nm_abundance = nm2midas(nm_df, strain_midas_df)

# Invader Check
# TODO @sunit: This is a temporary fix function;
# the data needs to be run through the comparison
# step again and changes needed there for a more final version
ic_df = read_ic_output(ic_pred_output, sample_names)
# ic_df.to_csv(f"{sample_week}.03_all_ic_predictions.csv", index=False)

# Transform Invader check Strain level outputs into Midas bucket terms
ic2midas_df = ic2midas(ic_df, strain_midas_df)

# Combine all data
all_cases = aggregate_across_tools(midas_df, nm2midas_df, ic2midas_df)
logging.info(
    f"Found {all_cases['unique_name'].nunique()} 'midas bucket - sample' combinations."
)
all_cases.to_csv(f"{output_folder}/{date_prefix}.all_cases.csv", index=False)

assert "Error" not in list(
    all_cases["tool_agreement"].unique()
), "NM-Midas agreement anomaly detected. Found a case that hasn't been accounted for."

unique_names_before_merger = len(
    set(midas_df["unique_name"])
    | set(nm2midas_df["unique_name"])
    | set(ic2midas_df["unique_name"])
)

unique_names_after_merger = len(set(all_cases["unique_name"]))

# assert (
#     unique_names_before_merger == unique_names_after_merger
# ), f"Expected total number of unique names to stay the same before ({unique_names_before_merger}) and after ({unique_names_after_merger}) the merger."

# Remove midas mismapped buckets
clean_midas_buckets, removed_buckets = remove_mismapped_midas_buckets(
    all_cases, linked_buckets, control_samples, actual_samples
)
del all_cases

# Check: all_cases.query("(midas_bucket == 'Bacteroides_cellulosilyticus_58046') and (sample_name == 'SCV2-Week8-Mouse10')")

logging.info(
    f"Will attempt to categorize {clean_midas_buckets['unique_name'].nunique()} 'midas bucket - sample' combinations."
)
clean_midas_buckets.to_csv(
    f"{output_folder}/{date_prefix}.clean_midas_buckets.csv", index=False
)

input_df = identify_inputs(clean_midas_buckets)
invader_df = identify_invaders(clean_midas_buckets)

assert (
    len(set(input_df["unique_name"]) & set(invader_df["unique_name"])) == 0
), "Overlap between input and invader calls found!"

already_identified = set(input_df["unique_name"]) | set(invader_df["unique_name"])
mixed_df = identify_mixed(clean_midas_buckets, already_identified)

already_identified = already_identified | set(mixed_df["unique_name"])

unclear_df = identify_unclear(clean_midas_buckets, already_identified)

all_categorized_df = pd.concat([input_df, invader_df, mixed_df, unclear_df])
# all_categorized_df.shape
logging.info(
    f"Successfully categorized {all_categorized_df['unique_name'].nunique()} 'midas bucket - sample' combinations for {all_categorized_df['midas_bucket'].nunique()} buckets."
)

logging.info(
    f"Adding strain contribution weighted abundances to 'midas bucket - sample' combinations."
)
add_strain_context_back(
    all_categorized_df, strain_weighted_bucket_nm_abundance, ic_df, nm_df
).to_csv(
    output_file, index=False,
)

# check_prediction_consistency(all_categorized_df)
# .to_csv(
#     f"{output_folder}/prediction_consistency.csv"
# )

logging.info(f"Making some very interesting plots ...")
for week in sample_weeks:
    plot_sample_names = list(sample_order.query("Week == @week")["sample_id"])
    week_df = all_categorized_df.query("sample_name in @plot_sample_names")
    logging.info(
        f"Final predictions breakdown for {week}:\n{week_df['Final_Prediction'].value_counts()}"
    )
    logging.info(
        f"Final predictions buckets for {week}:\n{week_df['midas_bucket'].nunique()}"
    )
    plot_relative_abundance_comparison(
        all_categorized_df,
        f"{output_folder}/{date_prefix}.{week}.imputed_nm_vs_midas_abundance",
        color="Final_Prediction",
    )

    plot_bucket_relative_abundance(
        all_categorized_df,
        sample_names,
        f"{output_folder}/{date_prefix}.{week}.midas_abundance_by_prediction",
    )

# combine midas, ninjamap and ic data
# take final input/invader/mixed/unclear calls
# report proportions for mixed buckets
midas_df.query("midas_bucket == 'Bacteroides_stercorirosoris_59710'").sort_values(
    ["count_reads"]
)
# visualize_predictions(df)
