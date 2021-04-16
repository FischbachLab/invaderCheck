#!/usr/bin/env python3

import sys
import pandas as pd
import logging
import boto3


def get_file_names(bucket_name, prefix, suffix="txt"):
    """
    Return a list for the file names in an S3 bucket folder.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (folder name).
    :param suffix: Only fetch keys that end with this suffix (extension).
    """
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    objs = response["Contents"]

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
    return shortlisted_files


def aggregate_ic_predictions(file_paths_list):
    list_of_dfs = [
        pd.read_csv(file_path.strip(), header=0) for file_path in file_paths_list
    ]

    return pd.concat(list_of_dfs).reset_index(drop=True)


def visualize_predictions(df):
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )

    # pred_paths_file = sys.argv[1]
    # out_prefix = sys.argv[2]
    # week = "W5"
    week = sys.argv[1]
    method = "majority"
    out_prefix = f"{week}_{method}"
    s3_bucket = "czbiohub-microbiome"
    s3_prefix = f"Sunit_Jain/scratch/immigrationCheck/output/v3_20201102/{method}"
    s3_suffix = "02d_stats.csv"

    all_pred_paths_list = get_file_names(s3_bucket, s3_prefix, s3_suffix)

    this_weeks_predictions = [
        f"s3://{s3_bucket}/{s3_path}"
        for s3_path in all_pred_paths_list
        if f"/{week}/" in s3_path
    ]
    logging.info(
        f"Aggregating data for {len(this_weeks_predictions)} invaderCheck predictions"
    )
    df = aggregate_ic_predictions(this_weeks_predictions)
    df.to_csv(f"{out_prefix}.03_predictions.csv", index=False)

    # visualize_predictions(df)
