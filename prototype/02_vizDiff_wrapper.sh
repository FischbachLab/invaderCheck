#!/bin/bash -x
# shellcheck disable=SC2086

set -euo pipefail

ORG_NAME=${1%/}
COMPARE_WITH="${2}"
S3_BASE_PATH="s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/01_vectors"
S3_OUTPUT_BASE_PATH=${4:-"s3://czbiohub-microbiome/Sunit_Jain/scratch/immigrationCheck/output/v3_20201102/diff_arrays"}
S3_OUTPUT_BASE_PATH=${S3_OUTPUT_BASE_PATH%/}

# Create comparison list
# bash 02a_create_comparison_list.sh ${ORG_NAME} ${COMPARE_WITH}
mkdir -p ${ORG_NAME}
aws s3 ls ${S3_BASE_PATH}/${ORG_NAME}/${ORG_NAME}_q20_id99_aln100_vectors/ | \
  awk -v s3path="${S3_BASE_PATH}/${ORG_NAME}/${ORG_NAME}_q20_id99_aln100_vectors" '{printf "%s/%s\n", s3path, $4}' > ${ORG_NAME}/all_vectors.s3paths.list

rm -rf ${ORG_NAME}/top.s3paths.list ${ORG_NAME}/bottom.s3paths.list 
for i in $(seq 1 17); do
        grep "W4M${i}\\." ${ORG_NAME}/all_vectors.s3paths.list >> ${ORG_NAME}/top.s3paths.list
        grep "${COMPARE_WITH}M${i}_" ${ORG_NAME}/all_vectors.s3paths.list >> ${ORG_NAME}/bottom.s3paths.list
done

# Patient and Community inoculum
{
  grep W4M6\\. ${ORG_NAME}/all_vectors.s3paths.list  
  grep W4M7\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M8\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M10\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M11\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M12\\. ${ORG_NAME}/all_vectors.s3paths.list  
  grep W4M14\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M15\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M16\\. ${ORG_NAME}/all_vectors.s3paths.list  
  grep W4M1\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M2\\. ${ORG_NAME}/all_vectors.s3paths.list
  grep W4M3\\. ${ORG_NAME}/all_vectors.s3paths.list  
} >> ${ORG_NAME}/top.s3paths.list 

{
  grep 'Pat1' ${ORG_NAME}/all_vectors.s3paths.list  
  grep 'Pat2' ${ORG_NAME}/all_vectors.s3paths.list  
  grep 'Pat3' ${ORG_NAME}/all_vectors.s3paths.list  
  grep 'Com' ${ORG_NAME}/all_vectors.s3paths.list 
} >> ${ORG_NAME}/bottom.s3paths.list 

num_top=$(wc -l ${ORG_NAME}/top.s3paths.list | awk '{print $1}')
num_bottom=$(wc -l ${ORG_NAME}/bottom.s3paths.list | awk '{print $1}')
if [[ num_top -eq num_bottom ]]; then
  paste ${ORG_NAME}/top.s3paths.list ${ORG_NAME}/bottom.s3paths.list > ${ORG_NAME}/compare.s3paths.list
else
  echo "[ERROR] Unequal number of items being compare. Can not do that yet ... Exiting."
  exit 1
fi

# Compare
python 02b_compute_strain_difference.py ${ORG_NAME} ${ORG_NAME}/compare.s3paths.list &> ${ORG_NAME}.02b.log

# Adjust 02b output location
OUTPUTDIR="${ORG_NAME}/${COMPARE_WITH}"
mkdir -p ${OUTPUTDIR}/prob_diff_arrays
mv ${ORG_NAME}/*.list ${OUTPUTDIR}/
mv ${ORG_NAME}*.npy ${OUTPUTDIR}/prob_diff_arrays/
mv ${ORG_NAME}.02b* ${OUTPUTDIR}/

aws s3 sync ${OUTPUTDIR} ${S3_OUTPUT_BASE_PATH}/${OUTPUTDIR}
rm -rf ${ORG_NAME:?}*