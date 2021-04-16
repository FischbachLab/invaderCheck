#!/usr/bin/env python3

##########################################################################################
####  THIS SCRIPT WAS USED TO GENERATE DEPTH vs COVERAGE FIGURES in the IN VIVO PAPER ####
##########################################################################################

import pandas as pd
import json
import math

import sys
import logging
import concurrent.futures
from tqdm import tqdm


def summarize_coverage(genome_name, sample_name, depth_csv):
    # depth_csv = "Acidaminococcus-sp-D21_q20_id100_aln100_vectors/W8M2.q20_id100_aln100.ref_depth.csv.gz"
    depth_df = pd.read_csv(depth_csv)
    depth_df["other_depth"] = depth_df["total_depth"] - depth_df["ref_depth"]
    genome_len, _ = depth_df.shape
    num_bases_covered_by_ref, _ = depth_df.query("ref_depth > 0").shape
    num_bases_covered_by_other, _ = depth_df.query("other_depth > 0").shape

    ref_percent_coverage = 100 * num_bases_covered_by_ref / genome_len
    other_percent_coverage = 100 * num_bases_covered_by_other / genome_len
    mean_ref_coverage_depth = depth_df["ref_depth"].mean()
    mean_other_coverage_depth = depth_df["other_depth"].mean()

    return {
        "genome_name": genome_name,
        "sample_name": sample_name,
        "ref_percent_coverage": ref_percent_coverage,
        "other_percent_coverage": other_percent_coverage,
        "mean_ref_coverage_depth": mean_ref_coverage_depth,
        "mean_other_coverage_depth": mean_other_coverage_depth,
    }


if __name__ == "__main__":
    args_file = sys.argv[1]
    depth_file = sys.argv[2]
    # args_file = "invaderCheck/mbfv1_w8.p100_a100.json"
    max_cores = 40
    with open(args_file, "r") as a:
        args = json.load(a)

    vector_df = pd.read_csv(args["vector_paths_file"])

    cov_summary = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        future = [
            executor.submit(
                summarize_coverage, row.Genome_Name, row.Sample_Name, row.Depth_Vector
            )
            for row in vector_df.itertuples()
        ]
        for f in tqdm(
            concurrent.futures.as_completed(future),
            ascii=True,
            desc="Summarizing Genome Coverage",
        ):
            cov_summary.append(f.result())

    pd.DataFrame(cov_summary).to_csv(depth_file, index=False)
