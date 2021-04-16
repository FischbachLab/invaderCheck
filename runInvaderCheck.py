#!/usr/bin/env python3
# Accept S3 dir for reference fasta
# Accept Parent S3 dir for BAMs
# Accept uniquely identifiable sample group
# Accept s3 output location.
# For each genome:
#   For each sample in this group:
#       get coverage distribution
#       write it to an appropriate s3 location

import concurrent.futures
import itertools
import logging
import boto3
import pandas as pd
import os
import sys
from pandas.io.parsers import read_csv
from tqdm import tqdm

from invaderCheck import genome_coverage_distribution_with_subsampling
from invaderCheck import compute_strain_difference
from invaderCheck import compare_distributions_wViz


def get_file_names(bucket_name, prefix, suffix="txt"):
    """
    Return a list for the file names in an S3 bucket folder.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (folder name).
    :param suffix: Only fetch keys that end with this suffix (extension).
    """
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    try:
        objs = response["Contents"]
    except KeyError as ke:
        logging.error(
            f"Path with bucket '{bucket_name}' and prefix '{prefix}' does not exist!"
        )
        raise(f"KeyError:{ke}")

    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            ContinuationToken=response["NextContinuationToken"],
        )
        objs.extend(response["Contents"])

    logging.info(f"Sifting through {len(objs)} files ...")

    shortlisted_files = list()
    if suffix == "":
        shortlisted_files = [obj["Key"] for obj in objs]
        total_size_bytes = sum([obj["Size"] for obj in objs])
    else:
        shortlisted_files = [obj["Key"] for obj in objs if obj["Key"].endswith(suffix)]
        total_size_bytes = sum(
            [obj["Size"] for obj in objs if obj["Key"].endswith(suffix)]
        )

    logging.info(
        f"Found {len(shortlisted_files)} files, totalling about {total_size_bytes/1e9:,.3f} Gb."
    )

    # return shortlisted_files
    return [f"s3://{bucket_name}/{file_path}" for file_path in shortlisted_files]


def get_bam_file(s3uri, sample_name, subfolder="bowtie2", suffix="bam"):
    bucket, file_prefix = declutter_s3paths(f"{s3uri}/{sample_name}/{subfolder}")
    return get_file_names(bucket, file_prefix, suffix="bam")[0]


def create_comparison_df(df, challenged_on):
    inoculum_query_samples = (
        df.query("Location == 'ex_vivo'")
        .reset_index(drop=True)
        .rename(columns={"sample_id": "query_sample_id"})
    )[["query_sample_id", "Week", "MouseNum", "MouseOrder"]]

    inoculum_weeks = set(inoculum_query_samples["Week"])

    base_samples = (
        df.query("Week == @challenged_on")
        .reset_index(drop=True)
        .rename(columns={"sample_id": "base_sample_id"})
    )[["base_sample_id", "Week", "MouseNum", "MouseOrder"]]

    # I'll only be able to use the mice who have a base to compare
    # pylint: disable=unused-variable
    selected_mice = sorted(base_samples["MouseNum"].unique())

    challenge_samples = (
        df.dropna(subset=["Challenge"])
        .query("MouseNum in @selected_mice")
        .reset_index(drop=True)
        .rename(columns={"sample_id": "query_sample_id"})
    )[["query_sample_id", "Week", "MouseNum", "MouseOrder"]]

    challenge_weeks = set(challenge_samples["Week"])

    all_query_samples = pd.concat([challenge_samples, inoculum_query_samples])
    compare_df = all_query_samples.merge(
        right=base_samples, on=["MouseNum", "MouseOrder"], suffixes=("_query", "_base")
    )

    return compare_df, challenge_weeks, inoculum_weeks


def setup_experiment(
    metadata, keep_locations=["ex_vivo", "gut"], challenged_on="Week4",
):
    # metadata = sample_metadata
    all_bam_paths_list = list()
    # setup_experiment(sample_metadata, s3_bam_dir, challenged_on="Week4")
    df = pd.read_csv(metadata, header=0).query("Location in @keep_locations")
    df["bam_file"] = df.apply(
        lambda x: get_bam_file(x.bam_location, x.sample_id), axis=1
    )
    all_bam_paths_list = sorted(df["bam_file"].unique())

    comparison_df, challenge_weeks, inoculum_weeks = create_comparison_df(
        df, challenged_on
    )

    comparisons = list()
    for week in challenge_weeks:
        if week in inoculum_weeks:
            continue
        # pylint: disable=unused-variable
        comparison_target_weeks = {week} | inoculum_weeks
        comparisons.append(
            comparison_df.query("Week_query in @comparison_target_weeks")
            .sort_values(["Week_query", "MouseOrder"], ascending=[False, True])
            .reset_index(drop=True)[["base_sample_id", "query_sample_id", "Week_query"]]
        )

    return all_bam_paths_list, comparisons, challenge_weeks


def declutter_s3paths(s3uri):
    s3path_as_list = s3uri.replace("s3://", "").rstrip("/").split("/")
    bucket = s3path_as_list.pop(0)
    prefix = "/".join(s3path_as_list)

    return bucket, prefix


def download_from_s3(s3_uri, local_dir):
    s3 = boto3.client("s3")
    bucket, file_obj = declutter_s3paths(s3_uri)
    local_file = f"{local_dir}/{os.path.basename(file_obj)}"
    if not os.path.exists(local_file):
        with open(local_file, "wb") as f:
            s3.download_fileobj(bucket, file_obj, f)

    return local_file


def upload_to_s3(s3_uri_dir, local_obj):
    s3 = boto3.client("s3")
    bucket, obj_dir = declutter_s3paths(s3_uri_dir)
    file_name = os.path.basename(local_obj)
    # with open(local_obj, "rb") as f:
    #     s3.upload_fileobj(f, bucket, f"{obj_dir}/{file_name}")
    s3.meta.client.upload_file(local_obj, bucket, f"{obj_dir}/{file_name}")

    return


def depth_vector_exists(genome, bam_file, min_qual, min_pid, min_paln):
    genome_name = os.path.splitext(os.path.basename(genome))[0]
    output_dir = f"{genome_name}_q{min_qual}_id{min_pid}_aln{min_paln}_vectors"
    file_name = os.path.basename(bam_file).split("_vs_")[0].split(".")[0]
    exp_vector_path = f"{output_dir}/{file_name}.q{min_qual}_id{min_pid}_aln{min_paln}.ref_depth.csv.gz"

    return os.path.exists(exp_vector_path)


def get_coverage_distribution(
    bam_s3_uri,
    fasta_list,
    local_tmp_dir,
    min_qual=20,
    min_pid=99,
    min_paln=100,
    subset_list=None,
):
    if not bam_s3_uri.endswith("bam"):
        return

    bam_file = f"{local_tmp_dir}/{os.path.basename(bam_s3_uri)}"
    bai_file = f"{bam_file}.bai"
    logging.info(f"Calculating coverage distribution {bam_file} ...")

    unprocessed_vectors = [
        depth_vector_exists(genome, bam_file, min_qual, min_pid, min_paln)
        for genome in fasta_list
    ]

    # download bam file and index, if needed.
    if not all(unprocessed_vectors):
        logging.info(f"Downloading {bam_file} ...")
        bam_file = download_from_s3(bam_s3_uri, local_tmp_dir)
        bai_file = download_from_s3(f"{bam_s3_uri}.bai", local_tmp_dir)

    # Get genome coverage for each genome in the bam file
    depth_files_list = [
        genome_coverage_distribution_with_subsampling.get_coverage_distribution(
            bam_file,
            fasta_file,
            min_qual=min_qual,
            min_pid=min_pid,
            min_paln=min_paln,
            subset_list=subset_list,
        )
        for fasta_file in fasta_list
    ]

    # delete bam file and index
    logging.info(f"Done processing {bam_file}.")
    if os.path.exists(bam_file):
        logging.info("Removing BAM files to save space.")
        os.remove(bam_file)
        os.remove(bai_file)

    return pd.DataFrame(depth_files_list)


def get_comparison_df(comparison_week, organism_df, all_challenge_weeks):
    # c = get_comparison_df(comparisons[0], organism_df)
    # c.to_csv("AKM_compare.csv", index=False)
    week_sets = set(comparison_week["Week_query"])
    assert (
        len(week_sets) == 2
    ), f"Comparison dataframe is malformed, contains the following sets: {week_sets}"

    week = all_challenge_weeks & week_sets

    df = (
        comparison_week.merge(
            how="left",
            right=organism_df,
            left_on="query_sample_id",
            right_on="Sample_Name",
        )
        .drop(["Sample_Name", "Genome_Name"], axis=1)
        .merge(
            how="left",
            right=organism_df,
            left_on="base_sample_id",
            right_on="Sample_Name",
            suffixes=("_query", "_base"),
        )
        .drop(["Sample_Name", "Genome_Name"], axis=1)
        .rename(columns={"Depth_Vector_query": "query", "Depth_Vector_base": "base"})
    )

    return week.pop(), df


def compute_depth_profiles(
    sample_metadata, s3_fasta_dir, vector_paths_file, local_tmp_dir="TEMP"
):
    # From the metadata file, get
    # 1. a list of bam files
    # 2. a list of df for each base week vs query week samples
    all_bam_paths_list, comparisons, challenge_weeks = setup_experiment(sample_metadata)

    # Download all genomes
    os.makedirs(local_tmp_dir, exist_ok=True)
    s3_fasta_bucket, s3_fasta_prefix = declutter_s3paths(s3_fasta_dir)
    s3_fasta_suffix = "fna"
    all_fasta_paths_list = get_file_names(
        s3_fasta_bucket, s3_fasta_prefix, s3_fasta_suffix
    )
    logging.info(f"Downloading {len(all_fasta_paths_list)} Genomes")
    local_fasta_files = [
        download_from_s3(fasta_s3_uri, local_tmp_dir)
        for fasta_s3_uri in tqdm(all_fasta_paths_list, ascii=True, desc="Genomes")
    ]
    genome_names = [
        os.path.splitext(os.path.basename(genome))[0] for genome in local_fasta_files
    ]

    _ = [
        os.makedirs(
            f"{genome_name}_q{min_qual}_id{min_pid}_aln{min_paln}_vectors",
            exist_ok=True,
        )
        for genome_name in genome_names
    ]

    vector_paths = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        future = [
            executor.submit(
                get_coverage_distribution,
                bam_file,
                local_fasta_files,
                local_tmp_dir,
                min_qual=min_qual,
                min_pid=min_pid,
                min_paln=min_paln,
                subset_list=None,
            )
            for bam_file in all_bam_paths_list
        ]
        for f in tqdm(
            concurrent.futures.as_completed(future),
            ascii=True,
            desc="Genome Coverage Distribution",
        ):
            vector_paths.append(f.result())

    vector_paths_df = pd.concat(vector_paths)
    # logging.info(vector_paths_df.shape)
    # logging.info(f"\n{vector_paths_df.head()}")
    vector_paths_df.to_csv(vector_paths_file, index=False)

    ## Upload vectors to S3
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
    #     future = [
    #         executor.submit(
    #             upload_to_s3,
    #             f"{s3_vector_output_dir}/{row.Genome_Name}",
    #             row.Depth_Vector,
    #         )
    #         for row in vector_paths_df.itertuples()
    #     ]
    #     for f in tqdm(concurrent.futures.as_completed(future), ascii=True, desc="Uploading depth profiles to S3"):
    #         _ = f.result()

    return genome_names, comparisons, challenge_weeks, vector_paths_df


def compute_differences_per_genome_per_week(week, genome, compare_df, cores, plot=True):
    num_comparisons, _ = compare_df.shape
    logging.info(
        f"\n*** Computing {num_comparisons} difference vectors for {genome} from reference week to {week} ***\n"
    )
    cores_per_job = min(cores, num_comparisons)
    strain_week_df = compute_strain_difference.compute_differences(
        week, genome, compare_df, cores=cores_per_job, plotting=plot
    )
    strain_week_df["Organism"] = genome
    strain_week_df["QueryWeek"] = week
    return strain_week_df


def parallelize_compute_differences(
    genome_names,
    comparisons,
    challenge_weeks,
    vector_paths_df,
    weekly_differences_filepath,
    cores,
    plot,
):
    # compute strain differences
    comparison_parameters = list()
    for genome in genome_names:
        organism_df = vector_paths_df.query("Genome_Name == @genome")
        for comparison_week_df in comparisons:
            # compare_df has 2 columns, 1 = "base" vector path (top), 2 = "query" vector path (bottom) in Mouse order
            challenge_week, compare_df = get_comparison_df(
                comparison_week_df, organism_df, challenge_weeks
            )
            comparison_parameters.append((challenge_week, genome, compare_df))

    weekly_diffs_df_list = list()
    num_parallel_jobs = 2
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_parallel_jobs
    ) as executor:
        future = [
            executor.submit(
                compute_differences_per_genome_per_week,
                week,
                genome,
                compare_df,
                cores=int(cores / num_parallel_jobs),
            )
            for week, genome, compare_df in comparison_parameters
        ]
    for f in tqdm(
        concurrent.futures.as_completed(future), ascii=True, desc="Compute Differences",
    ):
        weekly_diffs_df_list.append(f.result())

    weekly_differences_df = pd.concat(weekly_diffs_df_list)
    weekly_differences_df.to_csv(weekly_differences_filepath, index=False)
    return weekly_differences_df


def get_invader_check_predictions(
    week,
    genome,
    intermediate_files_metadata_df,
    control_term="PBS",
    method="majority",
    cutoff=0.002,
    cores=10,
    min_extreme_pos=2,
):
    df = intermediate_files_metadata_df.query(
        "(Organism == @genome) and (QueryWeek == @week)"
    ).reset_index(drop=True)

    output_folder_path = os.path.join(genome, week)

    # *.npy (column: diff_array_path) for control mice
    control_profiles = list(df.query("Challenge == @control_term")["diff_array_path"])
    # *.npy (column: diff_array_path) for challenged mice
    sample_profiles = list(df.query("Challenge != @control_term")["diff_array_path"])
    # *.ref_depth.csv.gz (column: query) for all mice
    init_profiles_df = df.rename(
        columns={"query_sample_id": "Query", "query": "S3Path"}
    )[["Query", "S3Path"]]

    comparison_df_filepath = compare_distributions_wViz.run(
        output_folder_path,
        genome,
        method,
        control_profiles,
        sample_profiles,
        init_profiles_df,
        cutoff=cutoff,
        cores=cores,
        invader_extreme_pos=min_extreme_pos,
    )

    df["Comparison_Stats"] = comparison_df_filepath

    return df


def detect_invaders(
    challenge_weeks,
    genome_names,
    intermediate_files_metadata_df,
    ic_pred_df_file,
    cores,
    control_term="PBS",
    method="majority",
    cutoff=0.002,
    min_extreme_pos=2,
):
    num_parallel_jobs = 2
    ic_pred_df_list = list()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_parallel_jobs
    ) as executor:
        future = [
            executor.submit(
                get_invader_check_predictions,
                week,
                genome,
                intermediate_files_metadata_df,
                cores=int(cores / num_parallel_jobs),
                control_term=control_term,
                method=method,
                cutoff=cutoff,
                min_extreme_pos=min_extreme_pos,
            )
            for week, genome in itertools.product(challenge_weeks, genome_names)
        ]
    for f in tqdm(
        concurrent.futures.as_completed(future), ascii=True, desc="Compare Differences",
    ):
        ic_pred_df_list.append(f.result())

    ic_pred_df = pd.concat(ic_pred_df_list)
    ic_pred_df.to_csv(ic_pred_df_file, index=False)
    return ic_pred_df


def aggregate_ic_predictions(file_paths_list):
    list_of_dfs = [
        pd.read_csv(file_path.strip(), header=0) for file_path in file_paths_list
    ]

    return pd.concat(list_of_dfs).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )
    cores = os.cpu_count()
    max_cores = int(cores * 0.90)  # use at most 90% of all CPU
    min_qual = 20
    min_pid = 99
    min_paln = 100
    invader_extreme_pos = 2
    make_plots = True
    control_metadata_term = "PBS"
    s3_fasta_dir = (
        "s3://czbiohub-microbiome/ReferenceDBs/NinjaMap/Index/SCv2_3_20201208/fasta"
    )
    s3_base_output = "s3://czbiohub-microbiome/Synthetic_Community/invaderCheck/SCv2_3_20201208/Mouse_Backfill_v2"
    sample_metadata = f"{s3_base_output}/00_metadata/mbfv2_metadata.csv"
    s3_vector_output_dir = f"{s3_base_output}/01_vectors"
    s3_diff_output_dir = f"{s3_base_output}/02_diff_arrays"
    s3_compare_output_dir = f"{s3_base_output}/03_ic_pred"
    local_tmp_dir = "TEMP"
    vector_paths_file = "vector_paths.csv"
    weekly_differences_file = "weekly_df.csv"
    ic_pred_df_file = "compare_df.csv"

    # Final Output
    weekly_agg_pred_file = "mbfv2_majority.ic_pred.csv"

    metadata_df = pd.read_csv(sample_metadata, header=0)

    # For each genome in each sample:
    #   Compute per nucleotide coverage and probability distributions
    (
        genome_names,
        comparisons,
        challenge_weeks,
        vector_paths_df,
    ) = compute_depth_profiles(
        sample_metadata, s3_fasta_dir, vector_paths_file, local_tmp_dir
    )

    # For each genome in each sample:
    #   Identify differences in per nucleotide probability distributions between reference (Week4)
    #   and query (Week5-8) samples
    if os.path.exists(weekly_differences_file):
        weekly_differences_df = pd.read_csv(weekly_differences_file, header=0)
    else:
        weekly_differences_df = parallelize_compute_differences(
            genome_names,
            comparisons,
            challenge_weeks,
            vector_paths_df,
            weekly_differences_file,
            cores=cores,
            plot=make_plots,
        )

    # Organize intermediate output file paths in a dataframe
    intermediate_files_metadata_df = weekly_differences_df.merge(
        right=metadata_df,
        how="left",
        left_on="query_sample_id",
        right_on="sample_id",
        suffixes=("_weekly_diff", "_metadata"),
    )

    # For each genome in each sample:
    #   Detect CCNs and compare ENDs to ascertain is the observed organism is part of the original
    #   input community or not
    # TEST: ["Week7"],
    # [genome_names[-1]],
    if os.path.exists(ic_pred_df_file):
        ic_pred_df = pd.read_csv(ic_pred_df_file, header=0)
    else:
        ic_pred_df = detect_invaders(
            challenge_weeks,
            genome_names,
            intermediate_files_metadata_df,
            ic_pred_df_file,
            cores=cores,
            control_term="PBS",
            method="majority",
            cutoff=0.002,
            min_extreme_pos=invader_extreme_pos,
        )

    ic_pred_df.dropna(subset=["Comparison_Stats"], inplace=True)

    # Aggregate by week
    # for week in challenge_weeks:
    weekly_agg_pred_df = pd.DataFrame()
    weekly_ic_pred_files = ic_pred_df["Comparison_Stats"].unique()
    weekly_agg_pred_df = aggregate_ic_predictions(weekly_ic_pred_files)
    weekly_agg_pred_df.to_csv(weekly_agg_pred_file, index=False)

    # import importlib
    # importlib.reload(compare_distributions_wViz)
