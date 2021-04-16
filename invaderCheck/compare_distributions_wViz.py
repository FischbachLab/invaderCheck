#!/usr/bin/env python3

import concurrent.futures

# import glob
import itertools
import logging
import os
import re
import sys
import math

import matplotlib
import numpy as np

# import numpy.ma as ma
import pandas as pd
from scipy import stats

matplotlib.use("agg")
import matplotlib.pyplot as plt


def ks_test(a, b):
    """
    Accepts two distributions a and b
    Returns: tuple (KS Distance and p-value)
    # Significant p-value means a and b belong to different distributions
    """
    a_prime = remove_nan(a.flatten())
    b_prime = remove_nan(b.flatten())
    if len(a_prime) == 0 or len(b_prime) == 0:
        return [np.nan] * 2
    return stats.ks_2samp(a_prime, b_prime)


def cos_sim(a, b):
    """calculate cosine distance between reference and given."""
    a_prime = a.flatten()
    b_prime = b.flatten()
    if len(a_prime) == len(b_prime):
        return np.dot(a_prime, b_prime) / (
            np.linalg.norm(a_prime) * np.linalg.norm(b_prime)
        )
    else:
        raise Exception("LengthError: Cos_Sim requires arrays to be of the same size.")


def positional_drift(base_array, query_array, cutoff=0.002):
    """
    Description:
        Identify positions that have a low deviation from the center in the base and
        compare those positions in the query. Return the number of positions that drifted
        in the query
    Accept: (numpy array1, numpy array2, float)
        base    diff array for the base
        query   diff array for the query
        cutoff  deviation cutoff from center
    Returns: (int1, int2, int3)
        int1: number of positions within cutoff in base
        int2: number of positions that can be compared between base and query
        int3: number of positions (out of int1) outside cutoff in query

    Derivation with KC:
    mouse1 = np.load("Bacteroides-uniformis-ATCC-8492.W4M1_vs_W8M1.prob_diff.npy")
    nan_pos_mouse1 = ~np.isnan(mouse1)
    mouse1_no_nans = mouse1[nan_pos_mouse1]
    mouse1_no_nans[abs(mouse1_no_nans) > 0.002].shape

    mouse7 = np.load("Bacteroides-uniformis-ATCC-8492.W4M7_vs_W8M7.prob_diff.npy")
    mouse7_no_nans = mouse7[nan_pos_mouse1]
    filter_pos_mouse7 = np.array(~np.isnan(mouse7_no_nans) & (abs(mouse1_no_nans) < 0.002))
    mouse7_filter = mouse7_no_nans[filter_pos_mouse7]
    mouse7_filter[abs(mouse7_filter) > 0.002].shape
    """
    assert base_array.shape == query_array.shape, "array size mismatch!"

    base = base_array.flatten()
    query = query_array.flatten()
    # Get index of positions that have non-NaN values in Base
    base_non_nan_pos_idx = ~np.isnan(base)

    # Filter out NaN positions
    base_no_nans = base[base_non_nan_pos_idx]
    # base_len = len(base_no_nans[abs(base_no_nans) > cutoff])
    base_conserved_len = len(base_no_nans[abs(base_no_nans) < cutoff])

    # if no positions are found inside the cutoff, there is nothing to compare
    if base_conserved_len == 0:
        query_comparable_pos = np.nan
        query_drift = np.nan
    else:
        # Select positions in Query where Base has non-NAN values
        query_no_nans = query[base_non_nan_pos_idx]

        # Number of positions that can be compared (w/o NaNs)
        query_comparable_pos = len(query_no_nans[~np.isnan(query_no_nans)])

        # Select positions that are not NaNs in Query
        # AND
        # absolute value of Base is less than cutoff
        query_filtered = query_no_nans[
            (~np.isnan(query_no_nans)) & (abs(base_no_nans) < cutoff)
        ]

        # Calc the number of common positions that have changed significantly in Query
        query_drift = len(query_filtered[abs(query_filtered) > cutoff])

    return (base_conserved_len, query_comparable_pos, query_drift)


def remove_nan(prob_array):
    return prob_array[~np.isnan(prob_array)]


def summary_stats(d, cutoff=0.002):
    """
    Accept: an n-dim numpy array
    Do: flatten array and remove np.nan values
    Return: dict of summary stats of array with keys:
        min,max,mean,median,q10,q90,mad,iqr
    """
    d_prime = remove_nan(d.flatten())
    d_prime_len = len(d_prime)
    if d_prime_len > 0:
        d_stats = {
            "Total_Length": len(d.flatten()),
            "Non_NA_Length": d_prime_len,
            "Non_Zero_Length": len(d_prime[d_prime != 0]),
            "Extreme_Positions": len(d_prime[abs(d_prime) == 1]),
            "Nucl_Within_Threshold": len(d_prime[abs(d_prime) < cutoff]),
            "Min": np.nanmin(d_prime),
            "Max": np.nanmax(d_prime),
            "Mean": np.nanmean(d_prime),
            "Std_Dev": np.nanstd(d_prime),
            "Variance": np.nanvar(d_prime),
            "Q10": np.nanquantile(d_prime, 0.1),
            "Median": np.nanmedian(d_prime),
            "Q90": np.nanquantile(d_prime, 0.9),
            "MAD": stats.median_absolute_deviation(d_prime, nan_policy="omit"),
            "IQR": stats.iqr(d_prime, nan_policy="omit"),
            "Skew": stats.skew(d_prime, nan_policy="omit"),
            "Kurtosis": stats.kurtosis(d_prime, nan_policy="omit"),
        }
    else:
        logging.info(
            "Not enough data points for reliable summary statistics. Adding placeholders ..."
        )
        nan_value = np.nan
        d_stats = {
            "Total_Length": len(d.flatten()),
            "Non_NA_Length": d_prime_len,
            "Non_Zero_Length": nan_value,
            "Extreme_Positions": nan_value,
            "Nucl_Within_Threshold": nan_value,
            "Min": nan_value,
            "Max": nan_value,
            "Mean": nan_value,
            "Std_Dev": nan_value,
            "Variance": nan_value,
            "Q10": nan_value,
            "Median": nan_value,
            "Q90": nan_value,
            "MAD": nan_value,
            "IQR": nan_value,
            "Skew": nan_value,
            "Kurtosis": nan_value,
        }
    summ = "; ".join([f"{k}:{d_stats[k]}" for k in sorted(d_stats.keys())])
    logging.info(f"Depth summary: {summ}")
    return d_stats


def summary_stats_select(d, cutoff=0.002):
    """
    Accept: an n-dim numpy array
    Do: flatten array and remove np.nan values
    Return: dict of summary stats of array with keys:
        min,max,mean,median,q10,q90,mad,iqr
    """
    d_prime = remove_nan(d.flatten())
    d_prime_len = len(d_prime)
    if d_prime_len > 0:
        d_stats = {
            "Total_Length": len(d.flatten()),
            "Non_NA_Length": d_prime_len,
            "Non_Zero_Length": len(d_prime[d_prime != 0]),
            "Extreme_Positions": len(d_prime[abs(d_prime) == 1]),
            "Nucl_Within_Threshold": len(d_prime[abs(d_prime) < cutoff]),
            "Skew": stats.skew(d_prime, nan_policy="omit"),
            "Kurtosis": stats.kurtosis(d_prime, nan_policy="omit"),
        }
    else:
        logging.info(
            "Not enough data points for reliable summary statistics. Adding placeholders ..."
        )
        nan_value = np.nan
        d_stats = {
            "Total_Length": len(d.flatten()),
            "Non_NA_Length": d_prime_len,
            "Non_Zero_Length": nan_value,
            "Extreme_Positions": nan_value,
            "Nucl_Within_Threshold": nan_value,
            "Skew": nan_value,
            "Kurtosis": nan_value,
        }
    summ = "; ".join([f"{k}:{d_stats[k]}" for k in sorted(d_stats.keys())])
    logging.info(f"Depth summary: {summ}")
    return d_stats


def compare_arrays_to_base(
    np_base_complete, query_array_path, selected_pos=None, cutoff=None
):
    query_dict = {}
    name_split = os.path.basename(query_array_path).split(".")
    query_name = name_split[1]
    np_query_complete = np.load(query_array_path)
    logging.info(f" ... {query_name}")

    if selected_pos is None:
        np_base = np_base_complete
        np_query = np_query_complete
    else:
        np_base = np_base_complete[selected_pos]
        np_query = np_query_complete[selected_pos]

    ks_dist, ks_p_val = ks_test(np_base, np_query)
    cosine_sim = cos_sim(np_base, np_query)
    base_conserved_len, query_comparable_pos, query_drift = positional_drift(
        np_base, np_query, cutoff=cutoff
    )
    stats = summary_stats(np_query, cutoff=cutoff)
    query_dict = {
        "query_name": query_name,
        "ks_dist": ks_dist,
        "ks_p_val": ks_p_val,
        "base_conserved_pos": base_conserved_len,
        "query_comparable_pos": query_comparable_pos,
        "query_conserved_pos_drift": query_drift,
        "cosine_similarity": cosine_sim,
    }

    query_dict.update(stats)
    return query_dict


def compare_all(
    query_array_paths, base_array_paths, selected_pos=None, cores=None, cutoff=None
):
    df_dict_array = list()
    num_profiles = len(query_array_paths)
    logging.info(f"{num_profiles} profiles found.")
    # processed = dict()
    for outer_idx, base_array in enumerate(base_array_paths):
        # logging.info(f"[{outer_idx+1}/{num_profiles}] Starting with {base_array}")
        name_split = os.path.basename(base_array).split(".")
        org_name = name_split[0]
        base_name = name_split[1]

        np_base = np.load(base_array)
        # query_dict = {}
        logging.info(f"[{outer_idx+1}/{num_profiles}] Comparing {base_name} with ...")

        # Parallelize comparisons to base
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            future = [
                executor.submit(
                    compare_arrays_to_base, np_base, query_path, selected_pos, cutoff
                )
                for query_path in query_array_paths
            ]
            for f in concurrent.futures.as_completed(future):
                result = f.result()
                # logging.info(
                #     f"Saving results for {base_name} vs {result['query_name']}"
                # )
                result.update({"base_name": base_name, "org_name": org_name})
                df_dict_array.append(result)

    return df_dict_array


def plot_diff_hist(ax, diff_vector, title):
    logging.info(f"Generating histogram for {title} ...")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.hist(diff_vector.flatten(), bins=np.arange(-1, 1.01, 0.01))
    ax.set_xlim([-1.15, 1.15])
    ax.set_rasterized(True)


def visualizing_differences(diff_vector, selected_pos, axs):
    title = os.path.basename(diff_vector).split(".")[1]
    loaded_diff = np.load(diff_vector)
    base_name = title.split("_vs_")[0]
    query_name = title.split("_vs_")[1]

    masked_diff = loaded_diff[selected_pos]
    plot_diff_hist(axs, masked_diff, title)
    stat_dict = summary_stats_select(masked_diff)
    stat_dict.update({"Base": base_name, "Query": query_name, "Sample": title})
    return stat_dict


def within_limits(array, cutoff):
    """
    returns a boolean array as int(0 or 1)
    """
    logging.info(f"Loading {array} at {cutoff}")
    loaded = np.load(array)
    with np.errstate(invalid="ignore"):
        # When comparing, nan always returns False.
        # return values as 0/1
        return (abs(loaded) < cutoff).astype(int)


def get_diff_stdev(array_paths, selected_idx):
    # Add absolute values of differences
    cumulative_diff = np.array(
        [subset_on_indices(abs(np.load(array)), selected_idx) for array in array_paths]
    )
    cum_diff = np.nansum(cumulative_diff, axis=0)
    return np.nanstd(cum_diff)


def select_conserved_positions(arrays, method="majority", cutoff=0.002):
    """
    Description:
        Given a list of diff prob arrays and a absolute cutoff value,
        return a vector based on the method containing the:
        - intersection of all positions within the cutoff; or
        - union of all positions within the cutoff; or
        - most prevelant (3/5) all positions within the cutoff

    Return: np.ndarray (boolean)
    """
    num_arrays = len(arrays)
    logging.info(
        f"Received {num_arrays} arrays, with method {method} and cutoff {cutoff}"
    )
    ## for each array:
    ##   get an int boolean (0/1) of indices within abs(cutoff)
    ##   sum each base value at each position across all arrays
    agg_array = np.sum([within_limits(array, cutoff) for array in arrays], axis=0)

    # Make sure I have the axis right.
    # If any position's total is greater than the number of arrays,
    # Means I'm adding along the wrong axis
    assert np.sum((agg_array > num_arrays).astype(int)) == 0, "Wrong axis!"
    selected_idx = []
    if method == "majority":
        if num_arrays % 2 == 1:
            majority_at = (num_arrays - 1) / 2
        else:
            majority_at = (num_arrays) / 2
        # majority rule
        ##   if value > majority --> majority of arrays have this base at this position within the cutoff.
        selected_idx = (agg_array > majority_at).astype(bool)
    elif method == "intersection":
        # intersection index
        ##   if value == num_arrays --> all arrays have this base at this position within the cutoff.
        selected_idx = (agg_array == num_arrays).astype(bool)
    elif method == "union":
        # union index
        ##   if value > 0 --> at least one array has this base at this position within the cutoff.
        selected_idx = (agg_array > 0).astype(bool)

    selected_idx_num = np.sum(selected_idx)
    logging.info(
        f"Method: {method}, Shape:{selected_idx.shape}, Num_Positions:{selected_idx_num}"
    )

    return selected_idx


def get_conserved_positions(arr_paths, method, cutoff=0.002, cores=None):
    """take a list of N arrays
    find conserved positions based on all combinations of N-1 arrays
    assign scores to each conserved position based on it's occurance in each combination


    Args:
        arr_paths ([type]): [description]
        method ([type]): [description]
        cutoff (float, optional): [description]. Defaults to 0.002.
        cores ([type], optional): [description]. Defaults to None, meaning "all available".

    Returns:
        [type]: [description]
    """
    num_arr = len(arr_paths)
    arr_combinations = list()
    logging.info(f"Found {num_arr} arrays")

    # based on all combinations of N-1 arrays
    index_combinations = list(itertools.combinations(range(num_arr), num_arr - 1))
    for idx_combo in index_combinations:
        arr_combinations.append([arr_paths[i] for i in idx_combo])
    logging.info(f"Created {len(arr_combinations)} combinations")

    # find conserved positions across N-1 arrays
    all_selected_indices = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        future = [
            executor.submit(select_conserved_positions, arr_sub, method, cutoff)
            for arr_sub in arr_combinations
        ]
        for f in concurrent.futures.as_completed(future):
            selected_index = f.result()
            all_selected_indices.append(selected_index.astype(int))

    selected_idx, selected_idx_err = pick_best_set(all_selected_indices, arr_paths)

    if selected_idx is None:
        logging.info(
            "Could not find any stable positions in controls, possibly due to lack of breadth or depth of coverage. Exiting ..."
        )
        raise CoverageError(
            "NotEnoughCoverage",
            "Could not find any stable positions in controls, possibly due to lack of breadth or depth of coverage.",
        )

    selected_idx_num = np.sum(selected_idx)
    logging.info(
        f"[Leave One Out] Method: {method}, Shape:{selected_idx.shape}, Num_Positions:{selected_idx_num}, Standard Error={selected_idx_err}"
    )

    return selected_idx.astype("bool")


def pick_best_set(selected_idx_array, diff_paths):
    """Pick the array set with the lowest std err (std / sqrt(sel_pos_len))
        if not std err can be calculated, pick longest.

    Args:
        selected_idx_array (list): list of geneome X 4 (nucleotides) numpy arrays with boolean values for selected positions
        diff_paths (list):  paths to numpy diff arrays of same shape as each array in selected_idx_array

    Returns:
        tuple: (selected_idx, selected_idx_stderr)
    """

    # apply selected idx and calculate std err across all samples.
    all_std = [
        get_diff_stdev(diff_paths, selected_idx) for selected_idx in selected_idx_array
    ]

    all_lengths = [np.sum(a) for a in selected_idx_array]
    by_std_err = np.full_like(all_std, np.nan)
    by_len = np.zeros_like(all_std)
    for idx, l in enumerate(all_lengths):
        if (l > 0) and (all_std[idx] > 0):
            by_std_err[idx] = all_std[idx] / np.sqrt(l)
        elif (l > 0) and (all_std[idx] == 0):
            by_len[idx] = l
        elif l == 0:
            continue

    best_pick_idx = np.nan
    if not np.all(np.isnan(by_std_err)):
        best_pick_idx = np.nanargmin(by_std_err)
    elif any(by_len > 0):
        best_pick_idx = np.nanargmax(by_len)

    # logging.info(f"Best Pick Index: {best_pick_idx}")
    if np.isnan(best_pick_idx):
        return (None, None)
    else:
        selected_arr_stderr = by_std_err[best_pick_idx]
        selected_arr_length = all_lengths[best_pick_idx]
        logging.info(
            f"Selected indices have a length of {selected_arr_length} and stderr of {selected_arr_stderr}"
        )
        logging.info(
            f"Other Lengths were: {','.join([str(l) for i, l in enumerate(all_lengths) if i != best_pick_idx])}"
        )
        logging.info(
            f"Other StdErrs were: {','.join([str(se) for i, se in enumerate(by_std_err) if i != best_pick_idx])}"
        )
        return (selected_idx_array[best_pick_idx], selected_arr_stderr)


def read_list_file(list_file):
    with open(list_file) as files_list:
        paths = [filepath.rstrip("\n") for filepath in files_list]
    return paths


def infer_week_and_source(sample_name):
    week = np.nan
    source = np.nan

    source_format = r"W\d+M\d+"
    m = re.compile(source_format)

    patient_format = r"Pat\d+"
    p = re.compile(patient_format)

    comm_format = r"Com\d+"
    c = re.compile(comm_format)

    if m.match(sample_name):
        week_fmt = re.compile(r"^W\d+")
        week = week_fmt.search(sample_name).group().replace("W", "")

        source_fmt = re.compile(r"M\d+")
        source = source_fmt.search(sample_name).group()
    elif p.match(sample_name):
        week = 0
        source = p.search(sample_name).group()
    elif c.match(sample_name):
        week = 0
        source = c.search(sample_name).group()

    return week, source


def get_array_stats(a):
    array = a.flatten()
    (col_sum, col_mean, col_median, col_std) = [np.nan] * 4

    col_sum = np.nansum(array)
    if col_sum > 0:
        col_mean = np.nanmean(array)
        col_median = np.nanmedian(array)
        col_std = np.nanstd(array)

    return col_sum, col_mean, col_median, col_std


def subset_on_indices(array, bool_idx):
    return np.where(bool_idx, array, np.full_like(array, np.nan))


def selected_pos_depths(query, depth_profile, informative_pos):
    # Find all positions where at least one nucleotide has a value
    selected_pos_idx = np.sum(informative_pos, axis=1) > 0
    selected_pos_len = len(selected_pos_idx)
    total_selected_pos = np.sum(selected_pos_idx)
    logging.info(f"Calculating depths for informative positions from {query} ...")

    # data_frame = pd.read_csv(
    #     depth_profile,
    #     header=0,
    #     usecols=["A", "C", "G", "T", "total_depth"],
    #     dtype={"A": "int", "C": "int", "G": "int", "T": "int", "total_depth": "float",},
    # )

    nucl_depth, pos_depth = get_depth_vectors(depth_profile)

    # Nucleotides
    # n_depth_arr = subset_on_indices(nucl_depth, informative_pos)
    # logging.info(f"{nucl_depth.shape}, {informative_pos.shape}")
    (
        selected_nt_total,
        selected_nt_mean,
        selected_nt_median,
        selected_nt_std,
    ) = get_array_stats(nucl_depth[informative_pos])

    (
        unselected_nt_total,
        unselected_nt_mean,
        unselected_nt_median,
        unselected_nt_std,
    ) = get_array_stats(nucl_depth[~informative_pos])

    # Positions
    _, og_total_mean, og_total_median, og_total_std = get_array_stats(pos_depth)

    genome_len = len(pos_depth)
    assert (
        selected_pos_len == genome_len
    ), f"Arrays do not have the same shape ({selected_pos_len} vs {genome_len}). Can't compare."

    non_zero_depth_pos = pos_depth > 0
    bases_covered = np.sum(non_zero_depth_pos)
    genome_cov = 100 * bases_covered / float(genome_len)

    selected_pos_depth = pos_depth[selected_pos_idx]
    _, sel_total_mean, sel_total_median, sel_total_std = get_array_stats(
        selected_pos_depth
    )

    filled_selected_pos = selected_pos_depth > 0
    selected_positions_found = np.sum(filled_selected_pos)

    selected_positions_found_perc = np.nan
    if total_selected_pos > 0:
        selected_positions_found_perc = (
            100 * selected_positions_found / float(total_selected_pos)
        )

    # Of the positions that have nucl values (filled_selected_pos),
    # how many nucl values per position?
    nucl_per_pos = np.nan
    if selected_positions_found > 0:
        nucl_depth_binary_idx = nucl_depth > 0
        nucl_per_pos_array = np.sum(nucl_depth_binary_idx, axis=1)

        nucl_per_pos = np.sum(nucl_per_pos_array[selected_pos_idx]) / float(
            selected_positions_found
        )

    sel_coef_of_var = np.nan
    if sel_total_mean > 0:
        sel_coef_of_var = sel_total_std / sel_total_mean
    perc_genome_selected = 100 * selected_positions_found / float(genome_len)

    # logging.info(
    #     f"{genome_len}, {genome_cov}, {nucl_per_pos}, {total_selected_pos}, {selected_positions_cov}, "
    #     f"{selected_positions_cov_perc}, {perc_genome_selected}, {sel_coef_of_var}, "
    #     f"{sel_total_mean}, {sel_total_median}, {sel_total_std},"
    #     f"{og_total_mean}, {og_total_median}, {og_total_std}, "
    # )
    return {
        "Query": query,
        "S3Path": depth_profile,
        "Ref_Genome_Len": genome_len,
        "Ref_Genome_Cov": genome_cov,
        "Nucl_Per_Pos": nucl_per_pos,
        "Pos_Selected_Searched": total_selected_pos,
        "Pos_Selected_Found": selected_positions_found,
        "Pos_Perc_Selected_Found": selected_positions_found_perc,
        "Pos_Perc_Genome_Selected": perc_genome_selected,
        "Pos_Selected_Coef_of_Var": sel_coef_of_var,
        "Pos_Selected_Mean_Depth": sel_total_mean,
        "Pos_Selected_Median_Depth": sel_total_median,
        "Pos_Selected_Stdev_Depth": sel_total_std,
        "Pos_All_Genome_Mean_Depth": og_total_mean,
        "Pos_All_Genome_Median_Depth": og_total_median,
        "Pos_All_Genome_Stdev_Depth": og_total_std,
        "Nucl_Selected_CumSum_Depth": selected_nt_total,
        "Nucl_Selected_Mean_Depth": selected_nt_mean,
        "Nucl_Selected_Median_Depth": selected_nt_median,
        "Nucl_Selected_Stdev_Depth": selected_nt_std,
        "Nucl_Unselected_CumSum_Depth": unselected_nt_total,
        "Nucl_Unselected_Mean_Depth": unselected_nt_mean,
        "Nucl_Unselected_Median_Depth": unselected_nt_median,
        "Nucl_Unselected_Stdev_Depth": unselected_nt_std,
    }


def parallelize_selected_pos_raw_data(depth_paths_df, informative_pos, cores):
    assert (
        "Query" in depth_paths_df.columns
    ), "Missing required column in depth path file: 'Query'"
    assert (
        "S3Path" in depth_paths_df.columns
    ), "Missing required column in depth path file: 'S3Path'"

    list_of_results = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        future = [
            executor.submit(
                selected_pos_depths, row.Query, row.S3Path, informative_pos,
            )
            for row in depth_paths_df.itertuples()
        ]
        for f in concurrent.futures.as_completed(future):
            result = f.result()
            # logging.info(f"Saving results for {result['Query_name']}")
            list_of_results.append(result)

    return pd.DataFrame(list_of_results).set_index(["Query"])


def get_depth_vectors(depth_profile):
    data_frame = pd.read_csv(
        depth_profile,
        header=0,
        usecols=["A", "C", "G", "T", "total_depth"],
        dtype={"A": "int", "C": "int", "G": "int", "T": "int", "total_depth": "float",},
    )
    genome_size, _ = data_frame.shape
    nucl_depth_vector = np.zeros((genome_size, 4))
    total_depth_vector = np.zeros(genome_size)

    # per Position
    nucl_depth_vector = data_frame[["A", "C", "G", "T"]].to_numpy()
    total_depth_vector = data_frame[["total_depth"]].to_numpy()

    return nucl_depth_vector, total_depth_vector


def get_extreme_pos_depth(diff_vector, depth_profile, informative_pos, values=None):
    if values is None:
        values = [0, 1]

    title = os.path.basename(diff_vector).split(".")[1]
    # base_name = title.split("_vs_")[0]
    query_name = title.split("_vs_")[1]

    logging.info(f"Calculating depth at extreme positions for {query_name}...")

    return_dict = {"Query": query_name}

    for value in values:
        value_dict = get_specific_diff_depth(
            diff_vector, depth_profile, informative_pos, value
        )
        return_dict.update(value_dict)

    return return_dict


def get_specific_diff_depth(diff_vector, depth_profile, informative_pos, value):
    (
        total_nucl_reads,
        mean_nucl_reads,
        sd_nucl_reads,
        total_reads,
        mean_reads,
        sd_reads,
    ) = [np.nan] * 6
    diff_vector_loaded = np.load(diff_vector)
    nucl_idx = abs(diff_vector_loaded) == value
    value_idx = np.sum(nucl_idx, axis=1) > 0

    if np.sum(value_idx) == 0:
        return {
            f"Pos_Read_Support_Total_at_{value}": total_reads,
            f"Pos_Read_Support_Mean_at_{value}": mean_reads,
            f"Pos_Read_Support_Stdev_at_{value}": sd_reads,
            f"Nucl_Read_Support_Total_at_{value}": total_nucl_reads,
            f"Nucl_Read_Support_Mean_at_{value}": mean_nucl_reads,
            f"Nucl_Read_Support_Stdev_at_{value}": sd_nucl_reads,
        }

    nucl_depth_vector, total_depth_vector = get_depth_vectors(depth_profile)

    # Selected Postions Nucleotide Depth
    selected_nucl_depth = nucl_depth_vector[nucl_idx & informative_pos]

    # selected_nucl_depth = np.sum(selected_nucl_depth_vector, axis=1)
    (total_nucl_reads, mean_nucl_reads, _, sd_nucl_reads) = get_array_stats(
        selected_nucl_depth
    )

    # Selected Postion Total Depth
    selected_pos_idx = np.sum(informative_pos, axis=1) > 0
    selected_pos_depth = total_depth_vector[value_idx & selected_pos_idx]

    # One read is counted once for each position, but may have been counted many times
    # over across multiple adjecent positions
    total_reads = np.nansum(selected_pos_depth)
    if total_reads > 0:
        mean_reads = np.nanmean(selected_pos_depth)
        sd_reads = np.nanstd(selected_pos_depth)

    (total_reads, mean_reads, _, sd_reads) = get_array_stats(selected_pos_depth)

    return {
        f"Pos_Read_Support_Total_at_{value}": total_reads,
        f"Pos_Read_Support_Mean_at_{value}": mean_reads,
        f"Pos_Read_Support_Stdev_at_{value}": sd_reads,
        f"Nucl_Read_Support_Total_at_{value}": total_nucl_reads,
        f"Nucl_Read_Support_Mean_at_{value}": mean_nucl_reads,
        f"Nucl_Read_Support_Stdev_at_{value}": sd_nucl_reads,
    }


def save_predictions(
    stats_df,
    read_support_df,
    output_file,
    selected_pos_depth,
    prefix,
    num_extereme_pos_for_invader=2,
):
    selected_pos_depth.drop(["S3Path"], inplace=True, axis=1)
    output_cols = (
        [
            "Organism",
            "Sample",
            "Source",
            "Week",
            "Extreme_Positions",
            "Prediction",
            "missed_2_EP",
        ]
        + list(selected_pos_depth.columns)
        + list(read_support_df.columns)
    )
    stats_df = stats_df.join(
        selected_pos_depth, on="Query", how="left", rsuffix="_other"
    )
    stats_df = stats_df.join(read_support_df, on="Query", how="left", rsuffix="_other")

    stats_df["Organism"] = prefix
    stats_df["Week"], stats_df["Source"] = zip(
        *stats_df["Query"].apply(infer_week_and_source)
    )
    stats_df["Prediction"] = stats_df.apply(
        lambda row: get_extreme_prediction(
            row["Extreme_Positions"], num_extereme_pos_for_invader
        ),
        axis=1,
    )

    stats_df["missed_2_EP"] = stats_df.apply(
        lambda row: invader_probability(
            row["Pos_Selected_Found"],
            row["Pos_Selected_Searched"],
            num_extereme_pos_for_invader,
        ),
        axis=1,
    )

    # Update Input Predictions based on missed_2_EP 5% tolerance and Nucl_Read_Support_Total_at_0 > 1000
    stats_df["Prediction"] = stats_df.apply(
        lambda row: adjust_predictions(
            row["Prediction"], row["missed_2_EP"], row["Nucl_Read_Support_Total_at_0"]
        ),
        axis=1,
    )
    stats_df.to_csv(output_file, index=False, columns=output_cols)


def get_extreme_prediction(num_extreme_positions, min_extreme_pos=2):
    prediction = "Unclear"

    if np.isnan(num_extreme_positions):
        prediction = "Unclear"
    elif num_extreme_positions >= min_extreme_pos:
        prediction = "Invader"
    elif num_extreme_positions < min_extreme_pos:
        prediction = "Input"
    else:
        prediction = "Error"

    return prediction


def adjust_predictions(
    current_pred,
    missed_2_EP,
    cum_read_supp_0,
    missed_EP_tolerance=0.05,
    min_cum_read_support=1000,
):
    adj_pred = "Unclear"
    if current_pred.lower() == "invader":
        adj_pred = current_pred
    elif (  # Update Input Predictions based on:
        (current_pred.lower() == "input")
        #  missed_2_EP 5% tolerance and
        and (missed_2_EP < missed_EP_tolerance)
        #  Nucl_Read_Support_Total_at_0 > 1000
        and (cum_read_supp_0 >= min_cum_read_support)
    ):
        adj_pred = "Input"

    return adj_pred


# def prob_extreme_pos(exp_extreme_pos, fraction_min_depth):
#     # ln p = f*S*ln (1-N/S) ~ -f*S*N/S = -f*N
#     logP = -fraction_min_depth * exp_extreme_pos
#     return logP


# def pos_frac_combinations(fraction_min_depth, possible_extreme_pos):
#     # log(N!) = N * math.log(N) - N
#     # C = S!/[(f*S)!((1-f)*S)!]
#     log_factorial = lambda N: N * math.log(N) - N
#     logC = log_factorial(possible_extreme_pos) - (
#         log_factorial(fraction_min_depth * possible_extreme_pos)
#         + log_factorial((1 - fraction_min_depth) * possible_extreme_pos)
#     )

#     return logC


# def missed_invader_probability(
#     exp_extreme_pos, possible_extreme_pos, fraction_min_depth
# ):
#     logP = prob_extreme_pos(exp_extreme_pos, fraction_min_depth)
#     logC = pos_frac_combinations(fraction_min_depth, possible_extreme_pos)
#     try:
#         q = math.exp(logP + logC)
#     except OverflowError:
#         q = float("inf")
#     return q


def invader_probability(obs_cons_pos, exp_cons_pos, missed_extrm_pos):
    """
    If you have S conserved positions, and you actually observe O = (S-q) of
    them due to some having low coverage, then calculate the probability that
    you picked a set of O that avoid the N positions that could be extreme
    positions

    # S = expected conserved positions
    # O = observed conserved positions
    # q = conserved positions not found (S - O)
    # N = number of missed extreme pos
    # p = (S-N)!/S! * q!/(q-N)!
    # Using stirling approximation:
    # p = exp(log(S)*(-N)+N*N*1./S+log(factorial(q)*1./factorial(q-N)))

    Args:
        obs_cons_pos (int): observed conserved positions or "selected positions found"
        exp_cons_pos (int): expected conserved positions or "selected positions"
        missed_extrm_pos (int): calculate the probability for this many missing extreme pos

    Returns:
        float: probability for this many missing extreme positions
    """
    if (exp_cons_pos == 0) or np.isnan(exp_cons_pos):
        return np.nan

    if np.isnan(obs_cons_pos):
        return np.nan

    if np.isnan(missed_extrm_pos):
        return np.nan

    if exp_cons_pos < missed_extrm_pos:
        return 1

    # q = S - O
    missed_conserved_pos = exp_cons_pos - obs_cons_pos

    # # If we found all the positions, it's probably an Input.
    # if missed_conserved_pos == 0:
    #     return 0

    # Stirling approximation
    stirling_aprx = lambda N: N * math.log(N) - N if N > 1 else 1
    logP = (
        stirling_aprx(exp_cons_pos - missed_extrm_pos)
        - stirling_aprx(exp_cons_pos)
        + stirling_aprx(missed_conserved_pos)
        - stirling_aprx(missed_conserved_pos - missed_extrm_pos)
    )
    try:
        p = math.exp(logP)
    except OverflowError:
        p = float("inf")

    # This is a probability, it can't be greater than 1
    # This if condition manages edge cases like the following:
    # >>> invader_probability(2,2,2)
    # 5.021384230796916
    # >>> invader_probability(3,3,2)
    # 2.022153704931268
    if p > 1:
        p = 1

    return p


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class CoverageError(Error):
    """Exception raised possibly due to lack of breadth or depth of coverage

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class InfoError(Error):
    """Exception raised because we could not find any informative positions from the control samples."

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def run(
    output_folder_path,
    prefix,
    method,
    control_array_paths,
    query_array_paths,
    depth_profiles_df,
    cutoff=0.002,
    cores=10,
    invader_extreme_pos=2,
):
    assert method.lower() in ["intersection", "union", "majority"], logging.error(
        f"Method '{method}' not recognized. Exiting"
    )

    comparison_df_filepath = f"{output_folder_path}/{prefix}.02d_stats.csv"
    hist_plot = f"{output_folder_path}/{prefix}.02d_hist_plot.png"

    if os.path.exists(comparison_df_filepath) and os.path.exists(hist_plot):
        return comparison_df_filepath

    try:
        informative_pos = get_conserved_positions(
            control_array_paths, method=method, cutoff=cutoff, cores=cores
        )
    except CoverageError as nec:
        logging.info(nec)
        return
    # np.save(f"{prefix}.{method}.informative_pos.npy", informative_pos)

    pos_depth_df = parallelize_selected_pos_raw_data(
        depth_profiles_df, informative_pos, cores=cores
    )

    try:
        if np.sum(informative_pos) == 0:
            logging.info(
                "Cannot find any informative positions from the control samples. Exiting."
            )
            # sys.exit(0)
    except InfoError as ie:
        logging.info(ie)
        return

    logging.info(
        f"Found {len(control_array_paths)} control profiles and {len(query_array_paths)} query profiles"
    )

    # Setup matplotlib figures
    all_array_paths = control_array_paths + query_array_paths

    # Parallelize extreme position read support (depth) calculations for each mouse
    read_support_list = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        future = [
            executor.submit(
                get_extreme_pos_depth,
                diff_vector,
                depth_profiles_df["S3Path"][idx],
                informative_pos,
            )
            for idx, diff_vector in enumerate(all_array_paths)
        ]
        for f in concurrent.futures.as_completed(future):
            result = f.result()
            # logging.info(f"Saving results for {result['Query_name']}")
            read_support_list.append(result)

    extreme_read_support_df = pd.DataFrame(read_support_list).set_index(["Query"])

    num_comparisons = len(all_array_paths)
    logging.info(f"Making {num_comparisons} comparisons ...")
    plot_height = int(num_comparisons * 10)
    plt.rcParams["figure.figsize"] = (45, plot_height)

    figure, hist_axs = plt.subplots(num_comparisons, 1, sharex=True, sharey=True)
    stats_df = pd.DataFrame(
        [
            visualizing_differences(diff_vector, informative_pos, hist_axs[idx])
            for idx, diff_vector in enumerate(all_array_paths)
        ]
    )

    save_predictions(
        stats_df,
        extreme_read_support_df,
        comparison_df_filepath,
        pos_depth_df,
        prefix,
        num_extereme_pos_for_invader=invader_extreme_pos,
    )

    logging.info(f"Saving histogram figure to disk as: {hist_plot}")
    figure.savefig(hist_plot, dpi=80, bbox_inches="tight")

    logging.info("All Done. Huzzah!")
    return comparison_df_filepath


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )

    prefix = sys.argv[1]
    method = sys.argv[2]
    # All files have samples in the same order
    control_profiles = sys.argv[3]
    sample_profiles = sys.argv[4]
    init_profiles = sys.argv[5]

    cutoff = 0.002
    cores = 10

    control_array_paths = read_list_file(control_profiles)
    query_array_paths = read_list_file(sample_profiles)
    depth_profiles_df = pd.read_table(
        init_profiles, header=None, names=["Query", "S3Path"]
    )
    output_folder_path = prefix

    run(
        output_folder_path,
        prefix,
        method,
        control_array_paths,
        query_array_paths,
        depth_profiles_df,
        cutoff=0.002,
        cores=10,
    )
