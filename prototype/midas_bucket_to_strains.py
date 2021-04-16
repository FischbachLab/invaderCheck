#!/usr/bin/env python3
"""
For each midas bucket, get at most 10 strains for a NM database.
"""

import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
import sys

# import shutil
import urllib.request as request
from contextlib import closing
from tqdm import tqdm
import boto3


def read_list_file(list_file):
    with open(list_file) as items_file:
        items_list = [i.rstrip("\n") for i in items_file]
    return items_list


def get_random_subset(df, max_genomes=10, seed=1712):
    num_genomes_available, _ = df.shape
    at_most = min(num_genomes_available, max_genomes)
    return df.sample(n=at_most, random_state=seed, replace=False)


def get_least_contigs_subset(df, max_genomes):
    num_genomes_available, _ = df.shape
    at_most = min(num_genomes_available, max_genomes)
    return df.sort_values(["contigs"]).head(at_most)


def declutter_s3paths(s3uri):
    s3path_as_list = s3uri.replace("s3://", "").rstrip("/").split("/")
    bucket = s3path_as_list.pop(0)
    prefix = "/".join(s3path_as_list)

    return bucket, prefix


def get_uris(genome_id, species_id, patric_ftp_path, patric_s3_path):
    ftp_link = f"{patric_ftp_path}/{genome_id}/{genome_id}.fna"
    s3_path = f"{patric_s3_path}/{species_id}___{genome_id}.fna"

    return ftp_link, s3_path


def get_genome(
    genome_id,
    species_id,
    patric_ftp_path,
    patric_s3_path,
    overwrite=False,
    max_retries=20,
    try_number=1,
    status="success",
):
    genome_acc, genome_ver = genome_id.split(".")

    ftp_link, s3_uri_file = get_uris(
        genome_id, species_id, patric_ftp_path, patric_s3_path
    )

    s3 = boto3.client("s3")
    bucket, obj_path = declutter_s3paths(s3_uri_file)
    not_exists = not (exists_on_s3(s3, bucket, obj_path))
    if overwrite or not_exists:
        # file_name = os.path.basename(obj_dir)
        try:
            with closing(request.urlopen(ftp_link)) as r:
                # with open(local_obj, "rb") as f:
                s3.upload_fileobj(r, bucket, f"{obj_path}")
                # s3.meta.client.upload_file(r, bucket, f"{obj_path}")
        except:
            logging.info(
                f"[{try_number}/{max_retries}]: Patric Genome ID '{genome_id}' failed."
            )
            if try_number < max_retries:
                status = "updated"
                try_number += 1
                genome_ver = int(genome_ver) + 1
                genome_id = f"{genome_acc}.{genome_ver}"
                logging.info(
                    f"[{try_number}/{max_retries}]: Trying next version: {genome_id}"
                )
                try:
                    uri_dict = get_genome(
                        genome_id,
                        species_id,
                        patric_ftp_path,
                        patric_s3_path,
                        overwrite=False,
                        max_retries=20,
                        try_number=try_number,
                        status=status,
                    )
                except:
                    return None

                # logging.info(f"{os.path.basename(s3_uri_file)} failed.")
                return uri_dict
            else:
                status = "failed"

    # logging.info(f"{os.path.basename(s3_uri_file)} downloaded.")
    return {
        "genome_id": genome_id,
        "species_id": species_id,
        "ftp_link": ftp_link,
        "s3_path": s3_uri_file,
        "status": status,
        "retries": try_number,
    }


def exists_on_s3(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except:
        return False
    return True


##############################################################################
##############################################################################
##############################################################################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )
    seed = 1712
    max_expansion = 10
    max_cores = 3
    output_name = sys.argv[1]
    # bucket_list_file = f"data/{mouse}_bucket.list"
    bucket_list_file = sys.argv[2]
    # output_dir = f"/Users/sunit.jain/Research/Sunit/Midas2NM/midas_bucket_expansion/mbfv2/{mouse}"
    output_dir = f"/Users/sunit.jain/Research/Sunit/Midas2NM/midas_bucket_expansion/mbfv1/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = f"{output_dir}/{output_name}.midas_strains"

    patric_ftp_path = "ftp://ftp.patricbrc.org/genomes"
    patric_s3_path = "s3://czbiohub-microbiome/ReferenceDBs/Midas/patric_fna"
    # Columns: species_id	rep_genome	count_genomes
    # midas_species_info_file = (
    #     "s3://czbiohub-microbiome/ReferenceDBs/Midas/v1.2/metadata/species_info.txt"
    # )
    midas_genome_info_file = (
        "s3://czbiohub-microbiome/ReferenceDBs/Midas/v1.2/metadata/genome_info.txt"
    )

    bucket_list = read_list_file(bucket_list_file)
    # Test
    # bucket_list = [
    #     "Bacteroides_cellulosilyticus_58046",
    #     "Bacteroides_vulgatus_57955",
    #     "Subdoligranulum_sp_62068",
    # ]

    # Columns: genome_id	genome_name	rep_genome	length	contigs	species_id
    midas_genome_info = pd.read_table(midas_genome_info_file, header=0).query(
        "species_id in @bucket_list"
    )
    num_genomes, _ = midas_genome_info.shape
    logging.info(f"{len(bucket_list)} buckets expanded to {num_genomes} genomes ...")

    # Definitely get the rep genome.
    logging.info("Selecting representative genomes for each midas bucket ...")
    rep_genomes = (
        midas_genome_info.query("rep_genome == 1")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # If number of genomes is greater than max_expansion, get:
    # - a random subset; OR
    # logging.info(
    #     f"Limiting 'midas bucket to genome' expansion to {max_expansion} genomes by random sampling ..."
    # )
    # unrep_genomes = (
    #     midas_genome_info.query("rep_genome == 0")
    #     .groupby(["species_id"])
    #     .apply(lambda x: get_random_subset(x, max_expansion, seed=seed))
    #     .drop_duplicates()
    #     .reset_index(drop=True)
    # )

    # - subset with least num of contigs; OR
    logging.info(
        f"Limiting 'midas bucket to genome' expansion to {max_expansion} genomes by least number of contigs ..."
    )
    unrep_genomes = (
        midas_genome_info.query("rep_genome == 0")
        .groupby(["species_id"])
        .apply(lambda x: get_least_contigs_subset(x, max_expansion - 1))
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # TODO: - subset with least difference in length from rep;
    # of max_expansion genomes.

    selected_genomes = pd.concat([rep_genomes, unrep_genomes]).reset_index(drop=True)
    selected_genomes.to_csv(f"{output_prefix}.from_midas.csv", index=False)
    num_genomes_to_download, _ = selected_genomes.shape
    logging.info(
        f"Will download {num_genomes_to_download} genomes for {len(bucket_list)} buckets ..."
    )

    # Test
    # get_genome(
    #     genome_id="411479.11",
    #     species_id="Bacteroides_uniformis_57318",
    #     patric_ftp_path=patric_ftp_path,
    #     patric_s3_path=patric_s3_path,
    #     overwrite=False,
    #     max_retries=20,
    #     status="success",
    # )

    dict_list = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        future = [
            executor.submit(
                get_genome,
                str(row.genome_id),
                row.species_id,
                patric_ftp_path=patric_ftp_path,
                patric_s3_path=patric_s3_path,
            )
            for row in selected_genomes.itertuples()
        ]
        for f in tqdm(
            concurrent.futures.as_completed(future),
            ascii=True,
            desc="Downloading Genomes from PATRIC",
        ):
            dict_list.append(f.result())

    downloaded_df = pd.DataFrame(dict_list)
    downloaded_df.to_csv(f"{output_prefix}.downloaded.csv", index=False)

    logging.info("Download summary:")
    logging.info(f"\n{downloaded_df['status'].value_counts()}")
    logging.info("All done! Huzzah!!")

