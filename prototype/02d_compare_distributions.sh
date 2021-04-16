#!/bin/bash -x
# shellcheck disable=SC2086
set -euo pipefail

ORG_NAME="${1}"
COMPARE_WITH="${2}"
METHOD=${3:-"majority"}
METHOD=${METHOD,,}
S3_BASE_PATH="s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck"
S3_VECTOR_PATH="${S3_BASE_PATH}/01_vectors/${ORG_NAME}/${ORG_NAME}_q20_id99_aln100_vectors"
S3_NPY_BASE_PATH=${4:-"${S3_BASE_PATH}/output/v3_20201102/diff_arrays"}
S3_OUTPUT_BASE_PATH=${5:-"${S3_BASE_PATH}/output/v3_20201102/${METHOD}"}

OUTPUTDIR=${ORG_NAME}/${COMPARE_WITH}
aws s3 sync --quiet ${S3_NPY_BASE_PATH}/${OUTPUTDIR}/prob_diff_arrays ${OUTPUTDIR}/prob_diff_arrays

# Input vs Invader calls
rm -rf ${ORG_NAME}/control_profiles.for_2d.list ${ORG_NAME}/samples_profiles.for_2d.list ${ORG_NAME}/init_profiles.s3paths.list
for i in $(seq 1 5); do
    find ${OUTPUTDIR}/prob_diff_arrays/ -name "*${COMPARE_WITH}M${i}.prob_diff.npy" >> ${ORG_NAME}/control_profiles.for_2d.list
    echo -e "${COMPARE_WITH}M${i}\t${S3_VECTOR_PATH}/${COMPARE_WITH}M${i}_vs_db_${ORG_NAME}.coord_sorted.q20_id99_aln100.ref_depth.csv.gz" >> ${ORG_NAME}/init_profiles.s3paths.list
done

for i in $(seq 6 17); do
    find ${OUTPUTDIR}/prob_diff_arrays/ -name "*${COMPARE_WITH}M${i}.prob_diff.npy" >> ${ORG_NAME}/samples_profiles.for_2d.list
    echo -e "${COMPARE_WITH}M${i}\t${S3_VECTOR_PATH}/${COMPARE_WITH}M${i}_vs_db_${ORG_NAME}.coord_sorted.q20_id99_aln100.ref_depth.csv.gz" >> ${ORG_NAME}/init_profiles.s3paths.list
done

for i in $(seq 1 3); do
    for j in $(seq 1 3); do
        echo -e "Pat${i}-${j}\t${S3_VECTOR_PATH}/Pat${i}-${j}.sortedByCoord.q20_id99_aln100.ref_depth.csv.gz" >> ${ORG_NAME}/init_profiles.s3paths.list
    done
done

for i in $(seq 1 3); do
    echo -e "Com${i}\t${S3_VECTOR_PATH}/Com${i}.sortedByCoord.q20_id99_aln100.ref_depth.csv.gz" >> ${ORG_NAME}/init_profiles.s3paths.list
done

{
    find ${OUTPUTDIR}/prob_diff_arrays/ -name "*Pat1*.prob_diff.npy" | sort
    find ${OUTPUTDIR}/prob_diff_arrays/ -name "*Pat2*.prob_diff.npy" | sort
    find ${OUTPUTDIR}/prob_diff_arrays/ -name "*Pat3*.prob_diff.npy" | sort
    find ${OUTPUTDIR}/prob_diff_arrays/ -name  "*Com*.prob_diff.npy" | sort
} >> ${ORG_NAME}/samples_profiles.for_2d.list

python 02d_compare_distributions_wViz.py \
    ${ORG_NAME} \
    ${METHOD} \
    ${ORG_NAME}/control_profiles.for_2d.list \
    ${ORG_NAME}/samples_profiles.for_2d.list \
    ${ORG_NAME}/init_profiles.s3paths.list &> ${ORG_NAME}.02d.log

rm -rf ${OUTPUTDIR}/prob_diff_arrays

# Move 02d outputs
mv ${ORG_NAME}/control_profiles.for_2d.list ${ORG_NAME}/samples_profiles.for_2d.list ${ORG_NAME}/init_profiles.s3paths.list ${OUTPUTDIR}/
mv ${ORG_NAME}.02* ${OUTPUTDIR}/

# Save output to S3
aws s3 sync --quiet ${ORG_NAME} \
    ${S3_OUTPUT_BASE_PATH}/${ORG_NAME}

# remove output dir
rm -rf ${ORG_NAME:?}/
