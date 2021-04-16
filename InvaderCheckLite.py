#!/usr/bin/env python3

##########################################################################################
####  THIS SCRIPT WAS USED TO GENERATE DEPTH vs COVERAGE FIGURES in the IN VIVO PAPER ####
##########################################################################################

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
import json

# from botocore.session import SubsetChainConfigFactory
import pandas as pd
import os

import sys

# from pandas.io.parsers import read_csv
from tqdm import tqdm
import botocore.exceptions
from invaderCheck import genome_coverage_distribution_with_subsampling

# from invaderCheck import compute_strain_difference
# from invaderCheck import compare_distributions_wViz


def get_file_names(bucket_name, prefix, suffix="txt"):
    """
    Return a list for the file names in an S3 bucket folder.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (folder name).
    :param suffix: Only fetch keys that end with this suffix (extension).
    """
    s3_client = boto3.client("s3")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except botocore.exceptions.ClientError:
        logging.error(f"Bucket={bucket_name}, Prefix={prefix}")
        # raise ce
        return None

    try:
        objs = response["Contents"]
    except KeyError as ke:
        logging.error(
            f"Path with bucket '{bucket_name}' and prefix '{prefix}' does not exist!"
        )
        raise ke

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
    s3uri = s3uri.rstrip("/")
    sample_name = sample_name.rstrip("/")

    bucket, file_prefix = declutter_s3paths(f"{s3uri}/{sample_name}/{subfolder}")
    return get_file_names(bucket, file_prefix, suffix="bam")[0]


def setup_experiment(
    metadata,
    keep_locations=["ex_vivo", "gut"],
    challenged_on="Week4",
):
    # metadata = sample_metadata
    all_bam_paths_list = list()
    # setup_experiment(sample_metadata, s3_bam_dir, challenged_on="Week4")
    df = metadata.query("Location in @keep_locations").dropna(subset=["bam_location"])
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


def compute_depth_profiles(
    sample_metadata,
    s3_fasta_dir,
    vector_paths_file,
    min_qual,
    min_pid,
    min_paln,
    local_tmp_dir="TEMP",
    max_cores=1,
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )
    cores = os.cpu_count()
    max_cores = int(cores * 0.90)  # use at most 90% of all CPU

    args_json = sys.argv[1]
    with open(args_json, "r") as j:
        args = json.load(j)

    sample_metadata = f"{args['s3_base_output']}/{args['sample_metadata_suffix']}"
    s3_vector_output_dir = (
        f"{args['s3_base_output']}/{args['s3_vector_output_dir_suffix']}"
    )
    s3_diff_output_dir = f"{args['s3_base_output']}/{args['s3_diff_output_dir_suffix']}"
    s3_compare_output_dir = (
        f"{args['s3_base_output']}/{args['s3_compare_output_dir_suffix']}"
    )

    # Final Output
    weekly_agg_pred_file = args["weekly_agg_pred_file"]
    filter_week = args["filter_week"]

    metadata_df = pd.DataFrame()
    if filter_week is not None:
        metadata_df = (
            pd.read_csv(sample_metadata, header=0)
            .query("Week in @filter_week")
            .dropna(subset=["bam_location"])
        )
    else:
        metadata_df = pd.read_csv(sample_metadata, header=0).dropna(
            subset=["bam_location"]
        )

    # For each genome in each sample:
    #   Compute per nucleotide coverage and probability distributions
    (
        genome_names,
        comparisons,
        challenge_weeks,
        vector_paths_df,
    ) = compute_depth_profiles(
        metadata_df,
        args["s3_fasta_dir"],
        args["vector_paths_file"],
        args["min_qual"],
        args["min_pid"],
        args["min_paln"],
        args["local_tmp_dir"],
        max_cores,
    )

