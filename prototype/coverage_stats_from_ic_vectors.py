#!/usr/bin/env python
# from tqdm import tqdm
import os
import pandas as pd
import concurrent.futures
import logging
import numpy as np
import math

# read vector depth df for a reference genome
# Calculate:
# - breadth of coverage: For a genome of length N bases how many nucleotides have at least 1 read support (regardless of a match).
# - mean depth of coverage: For a genome of length N, on average, how many reads cover each base in the genome?
# - also important to calculate other summary stats for depth, as all the depth might be restricted to small islands on the reference genome.

# Final Output:
# SampleName, Organism, mean depth of cov, % breadth of cov, genome length


def coverage_depth_stats(df, column):
    try:
        mean_depth = df[column].mean()
    except:
        mean_depth = np.nan
    return mean_depth


def coverage_breadth_stats(df, column):
    # 100 * number of rows greater than 0 / total rows
    try:
        perc_cov_breadth = 100 * df[column].gt(0).mean()
    except:
        perc_cov_breadth = np.nan
    return perc_cov_breadth


def series_stats(df, column):
    return (
        df[column].std(),
        df[column].var(),
        df[column].median(),
    )


def calculate_contamination_rate(df, ref_col, total_col):
    reads_with_ref = df[ref_col].sum()
    total_reads_aln = df[total_col].sum()

    c_rate = np.nan
    if total_reads_aln > 0:
        # c_rate = 1 - (reads_with_ref / total_reads_aln)
        c_rate = 1 - math.exp(math.log(reads_with_ref) - math.log(total_reads_aln))

    return c_rate


def calculate_coverage_stats(filepath):
    # logging.info(f"{filepath}")
    dirname, filename = os.path.split(filepath)
    sample_name = filename.split(".")[0]
    organism = os.path.basename(dirname).split("_")[0]

    try:
        df = pd.read_csv(filepath, header=0)
    except:
        logging.error(filepath)
        raise Exception({"sample_name": sample_name, "Organism": organism})

    genome_len, _ = df.shape

    ref_breadth_cov_perc = coverage_breadth_stats(df, column="ref_depth")
    ref_mean_depth_cov = coverage_depth_stats(df, column="ref_depth")
    (ref_depth_std, ref_depth_var, ref_depth_median,) = series_stats(
        df, column="ref_depth"
    )

    total_breadth_cov_perc = coverage_breadth_stats(df, column="total_depth")
    total_mean_depth_cov = coverage_depth_stats(df, column="total_depth")
    (total_depth_std, total_depth_var, total_depth_median,) = series_stats(
        df, column="total_depth"
    )

    contamination_rate = calculate_contamination_rate(df, "ref_depth", "total_depth")

    return {
        "sample_name": sample_name,
        "Organism": organism,
        "genome_length": genome_len,
        "contamination_rate": contamination_rate,
        "total_breadth_cov_perc": total_breadth_cov_perc,
        "total_depth_mean_cov": total_mean_depth_cov,
        "total_depth_std": total_depth_std,
        "total_depth_var": total_depth_var,
        "total_depth_median": total_depth_median,
        "ref_breadth_cov_perc": ref_breadth_cov_perc,
        "ref_depth_mean_cov": ref_mean_depth_cov,
        "ref_depth_std": ref_depth_std,
        "ref_depth_var": ref_depth_var,
        "ref_depth_median": ref_depth_median,
    }


def parallelize(perform, task_list, cores=None):
    if cores is None:
        cores = os.cpu_count() - 1

    list_of_dict = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(perform, task) for task in task_list}

        for f in concurrent.futures.as_completed(futures):
            list_of_dict.append(f.result())

    return pd.DataFrame(list_of_dict)


def read_list_file(list_file):
    with open(list_file) as items_file:
        items_list = [i.rstrip("\n") for i in items_file]
    return items_list


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )

    ## Test
    # paths_list = [
    #     "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2/01_vectors/Bacteroides-cellulosilyticus-DSM-14838-MAF-2_q20_id99_aln100_vectors/SCV2-Week8-Mouse1.q20_id99_aln100.ref_depth.csv.gz",
    #     "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2/01_vectors/Bacteroides-cellulosilyticus-DSM-14838-MAF-2_q20_id99_aln100_vectors/SCV2-Week8-Mouse2.q20_id99_aln100.ref_depth.csv.gz",
    #     "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2/01_vectors/Bacteroides-cellulosilyticus-DSM-14838-MAF-2_q20_id99_aln100_vectors/SCV2-Week8-Mouse3.q20_id99_aln100.ref_depth.csv.gz",
    # ]

    paths_listfile = "/home/ec2-user/efs/docker/Mouse_Backfill/immigrationCheck/q20_id99_aln100/db_SCv2_3/mbfv2_w8_ic_vectors.list"
    paths_list = read_list_file(paths_listfile)
    result_df = parallelize(calculate_coverage_stats, paths_list)
    result_df.to_csv(
        "/home/ec2-user/efs/docker/Mouse_Backfill/immigrationCheck/q20_id99_aln100/db_SCv2_3/ic_coverage_stats.csv",
        index=False,
    )
