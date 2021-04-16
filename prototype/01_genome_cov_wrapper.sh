#!/bin/bash -x
# USAGE:
# bash -x 01_genome_cov_wrapper.sh \
#     s3://czbiohub-microbiome/ReferenceDBs/NinjaMap/Index/SCv1_2_20200504/fasta/Acidaminococcus-fermentans-DSM-20731.fna \
#     s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/dbSCv1_2/Acidaminococcus-fermentans-DSM-20731 \
#     W5

# bash -x 01_genome_cov_wrapper.sh \
#     s3://czbiohub-microbiome/ReferenceDBs/NinjaMap/Index/SCv1_2_20200504/fasta/Bifidobacterium-catenulatum-DSM-16992.fna \
#     s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/dbSCv1_2/Bifidobacterium-catenulatum-DSM-16992 \
#     W5

# Results are uploaded to "s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/01_vectors/${GENOME_NAME}/"

set -euo pipefail

GENOME_S3PATH=${1}
BAM_S3DIR=${2%/}
SAMPLE_PREFIX="${3:-'W5'}"
VECTORS_S3_BASE="s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/01_vectors"

S3_BUCKET_BASE="s3://czbiohub-microbiome"

GENOME_NAME=$(basename "${GENOME_S3PATH}" .fna)
aws s3 cp ${GENOME_S3PATH} ${GENOME_NAME}.fna

DIST_SUMMARY=${GENOME_NAME}.${SAMPLE_PREFIX}.01b_summary.tsv
TMP_NAME=${SAMPLE_PREFIX}_${GENOME_NAME}

# Create a list of S3paths to BAM and BAM index files
aws s3 ls ${BAM_S3DIR} --recursive | \
    grep -E "${SAMPLE_PREFIX}" | \
    grep "bam" | \
    awk -v bucket="${S3_BUCKET_BASE}" '{printf "%s/%s\n", bucket, $4}' > ${TMP_NAME}.s3paths.list

# Download the BAM and BAM index files from the list
cat ${TMP_NAME}.s3paths.list | \
    parallel -j 3 "aws s3 cp {} ${TMP_NAME}/{/}"

# Execute the '01b_genome_coverage_distribution_with_subsampling.py' script.
find ${TMP_NAME} -name "*bam" | \
    grep -E "${SAMPLE_PREFIX}" | \
    parallel -j 3 "python 01_genome_coverage_distribution_with_subsampling.py {} ${GENOME_NAME}.fna" \
    &> ${DIST_SUMMARY}

# # If BAM is stored locally, comment the 'create list', 'download' 
# # and 'execute' code chunks 
# # and uncomment the following command
# find ${BAM_S3DIR} -name "${SAMPLE_PREFIX}*bam" | \
#     parallel -j 3 "python 01_genome_coverage_distribution_with_subsampling.py {} ${GENOME_NAME}.fna" \
#     &> ${DIST_SUMMARY}

rm ${GENOME_NAME}.fna
rm -rf ${TMP_NAME:?}/

mkdir -p ${GENOME_NAME}
mv ${SAMPLE_PREFIX}_${GENOME_NAME}* ${GENOME_NAME}/ || echo "No such file ${SAMPLE_PREFIX}_${GENOME_NAME}'*', Skipping ... "
mv ${GENOME_NAME}*tsv ${GENOME_NAME}/
mv ${GENOME_NAME}*_vectors ${GENOME_NAME}/

aws s3 sync ${GENOME_NAME} ${VECTORS_S3_BASE}/${GENOME_NAME}
rm -rf ${GENOME_NAME:?}/