#!/usr/bin/env python3
#
# Accepts subsampled fastq files, bam alignment of whole fastq and reference genome.
# How to subsample fastq files?
#  Use the bam file to get reads aligned to reference.
#  Subsample this set of reads to the desired coverage.

import gzip
import logging
import os
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import pysam
from Bio import SeqIO
from scipy import stats


class Contigs:
    genome_size = int(0)
    num_contigs = int(0)

    def __init__(self, record):
        Contigs.num_contigs += 1
        self.name = record.id
        self.sequence = str(record.seq)
        self.length = int(len(self.sequence))
        Contigs.genome_size += self.length

        self.ref_prob_array = np.zeros(self.length)
        self.ref_depth_array = np.zeros(self.length)
        self.ref_covered_bases = float(0)
        self.total_depth = float(0)
        self.ref_depth = float(0)

    def __hash__(self):
        return hash(str(self.sequence))

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both
        # x==y and x!=y being True at the same time
        return not (self == other)

    # def weighted_ref_depth(self):
    #     if self.ref_covered_bases == 0:
    #         return 0

    #     return Contigs.weighted_cal(self.ref_depth, self.ref_covered_bases, self.length)

    @staticmethod
    def weighted_cal(success, sample_size, total):
        return (success / float(sample_size)) * float(total)

    @classmethod
    def read_fasta(cls, fasta_file):
        return [
            Contigs(fasta_record) for fasta_record in SeqIO.parse(fasta_file, "fasta")
        ]


# def parallel_calculate_coverage(bamfile_name, contigs, min_qual, min_pid, min_paln, cores):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
#         future = [executor.submit(calculate_coverage_per_contig, bamfile_name, min_qual, min_pid, min_paln, contig) for contig in contigs]
#         for f in concurrent.futures.as_completed(future):
#             result = f.result()
#         ## When you don't wish to return anything, but monitor process for exceptions.
#         jobs = concurrent.futures.wait(future, return_when='FIRST_EXCEPTION')
#         if len(jobs.not_done) > 0:
#             print(f'Some ({len(jobs.not_done)}) exceptions occurred')
#             for job in jobs.not_done:
#                 print(f'{job.exception()}')
#             sys.exit(1)


def calculate_coverage(
    bamfile_name, contigs, read_set, min_qual, min_pid, min_paln, fh
):
    covered_contig_lengths = [
        calculate_coverage_per_contig(
            bamfile_name, read_set, min_qual, min_pid, min_paln, contig, fh
        )
        for contig in contigs
    ]

    return sum(covered_contig_lengths)


def calculate_coverage_per_contig(
    bamfile_name, read_set, minQual, thresh_pid, thresh_paln, contig, fh
):
    if len(read_set) == 0:
        check_read = partial(keep_read, min_pid=thresh_pid, min_paln=thresh_paln)
    else:
        check_read = partial(
            keep_read, min_pid=thresh_pid, min_paln=thresh_paln, read_set=read_set
        )
    with pysam.AlignmentFile(bamfile_name, mode="rb") as bamfile:
        # https://pysam.readthedocs.io/en/latest/api.html#pysam.AlignmentFile.count_coverage
        cov_counts = bamfile.count_coverage(
            contig.name,
            start=0,
            end=contig.length,
            read_callback=check_read,
            # quality_threshold is the minimum quality score (in phred) a base has to reach to be counted.
            quality_threshold=minQual,
        )
        contig_len = 0
        for i, ref_allele in enumerate(contig.sequence):
            base_cov = defaultdict(int)
            ref_prob = 0
            ref_depth = 0
            total_depth = 0

            ref_allele = str(ref_allele).upper()
            # if refseq has N, the prob is 'NA'
            if ref_allele in ["A", "T", "C", "G"]:
                base_cov["A"] = cov_counts[0][i]
                base_cov["C"] = cov_counts[1][i]
                base_cov["G"] = cov_counts[2][i]
                base_cov["T"] = cov_counts[3][i]
                total_depth = (
                    base_cov["A"] + base_cov["C"] + base_cov["G"] + base_cov["T"]
                )
                # prob 0 if no depth
                if total_depth > 0:
                    contig.total_depth += total_depth
                    ref_depth = base_cov[ref_allele]
                    ref_prob = float(ref_depth) / float(total_depth)
                    # if ref_depth > 0:
                    #     contig.ref_covered_bases += 1
                    #     contig.ref_depth += ref_depth
            else:
                ref_prob = np.nan
                ref_depth = np.nan
                # if ref_prob < 1:
                # Write output to a file, with columns as:
                # CHROM,POS,REF,A,C,G,T,ref_prob,ref_depth,total_depth
            fh.write(
                f"{contig.name},"
                f"{i},"
                f"{ref_allele},"
                f"{base_cov['A']},"
                f"{base_cov['C']},"
                f"{base_cov['G']},"
                f"{base_cov['T']},"
                f"{ref_prob},"
                f"{ref_depth},"
                f"{total_depth}"
                f"\n"
            )
            contig_len = i + 1
            # contig.ref_prob_array[i] = ref_prob
            # contig.ref_depth_array[i] = ref_depth
    return contig_len


def keep_read(aln, min_pid, min_paln, read_set=None):
    if (read_set is not None) and (aln.query_name not in read_set):
        return False

    edit_dist = dict(aln.tags)["NM"]
    aln_len = aln.query_alignment_length
    read_length = aln.infer_read_length()

    # MetaBAT: %ID is calculated from the CIGAR string and/or NM/MD fields
    # and == 100 * MatchedBases / (MatchedBases + Substituions + Insertions + Deletions)

    pid = 100 * (aln_len - edit_dist) / float(aln_len)
    p_aln_len = aln_len * 100 / float(read_length)

    return (min_pid <= pid <= 100) and (min_paln <= p_aln_len <= 100)


def distance_from_reference(prob_array):
    # remove NANs from array
    prob_array = remove_nan(prob_array)
    # create a unit array of same length representing reference.
    ref_array = np.ones_like(prob_array)

    return (
        root_mean_squared_error(ref_array, prob_array),
        cos_sim(ref_array, prob_array),
    )


def root_mean_squared_error(ref_array, prob_array):
    """ Calculate mean squared error given two arrays of same length """
    if len(ref_array) == len(prob_array):
        return np.sqrt(np.square(np.subtract(ref_array, prob_array)).mean())


def cos_sim(ref_array, prob_array):
    """calculate cosine distance between reference and given."""
    if len(ref_array) == len(prob_array):
        return np.dot(ref_array, prob_array) / (
            np.linalg.norm(ref_array) * np.linalg.norm(prob_array)
        )


def remove_nan(prob_array):
    return prob_array[~np.isnan(prob_array)]


def get_coverage_distribution(
    bam_file, genome, min_qual=20, min_pid=99, min_paln=100, subset_list=None
):
    genome_name = os.path.splitext(os.path.basename(genome))[0]
    output_dir = f"{genome_name}_q{min_qual}_id{min_pid}_aln{min_paln}_vectors"
    # os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(bam_file).split("_vs_")[0].split(".")[0]
    prefix = f"{output_dir}/{file_name}.q{min_qual}_id{min_pid}_aln{min_paln}"

    # ref_prob_array_name = f"{prefix}.ref_prob.npy"
    # ref_depth_array_name = f"{prefix}.ref_depth.npy"
    depth_file = f"{prefix}.ref_depth.csv.gz"
    if os.path.exists(depth_file):
        return {
            "Genome_Name": genome_name,
            "Sample_Name": file_name,
            "Depth_Vector": depth_file,
        }

    depth_file_handle = gzip.open(depth_file, "wt")
    depth_file_handle.write(
        f"contig_name,"
        f"pos,"
        f"ref_allele,"
        f"A,"
        f"C,"
        f"G,"
        f"T,"
        f"ref_prob,"
        f"ref_depth,"
        f"total_depth"
        f"\n"
    )

    contigs = Contigs.read_fasta(genome)

    read_set = set()
    if subset_list is not None:
        with open(subset_list, "r") as subset:
            for line in subset:
                read_set.add(line.strip())

    _ = calculate_coverage(
        bam_file, contigs, read_set, min_qual, min_pid, min_paln, depth_file_handle,
    )
    depth_file_handle.close()

    return {
        "Genome_Name": genome_name,
        "Sample_Name": file_name,
        "Depth_Vector": depth_file,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t[%(levelname)s]:\t%(message)s",
    )

    bam_file = sys.argv[1]
    genome = sys.argv[2]
    subset_list = None
    if len(sys.argv) > 3:
        subset_list = sys.argv[3]

    min_qual = 20  # quality_threshold is the minimum quality score (in phred) a base has to reach to be counted.
    min_pid = 99
    min_paln = 100

    get_coverage_distribution(
        bam_file, genome, min_qual=20, min_pid=99, min_paln=100, subset_list=None
    )
    sys.exit(0)

    # depth = (
    #     np.sum([contig.weighted_ref_depth() for contig in contigs])
    #     / Contigs.genome_size
    # )
    # coverage = (
    #     100
    #     * np.sum([contig.ref_covered_bases for contig in contigs])
    #     / Contigs.genome_size
    # )

    # # combine all prob arrays in order of contigs
    # genome_ref_prob = np.concatenate(
    #     [remove_nan(contig.ref_prob_array) for contig in contigs]
    # )
    # # np.save(ref_prob_array_name, genome_ref_prob)

    # genome_depth_prob = np.concatenate(
    #     [remove_nan(contig.ref_depth_array) for contig in contigs]
    # )
    # # np.save(ref_depth_array_name, genome_depth_prob)

    # depth_ent = stats.entropy(genome_depth_prob)
    # rmse, sim = distance_from_reference(genome_ref_prob)
    # print(f"{file_name}\t{depth}\t{coverage}\t{depth_ent}\t{rmse}\t{sim}")
