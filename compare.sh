while getopts u:p:d:l: flag
do
    case "${flag}" in
        u) user_id=${OPTARG};;
        p) project_id=${OPTARG};;
        d) dataset_id=${OPTARG};;
        l) pipeline_id=${OPTARG};;
    esac
done 

echo "user_id: $user_id";
echo "project_id: $project_id";
echo "dataset_version_id: $dataset_id";
echo "pipeline_id: $pipeline_id";

# export USER_ID=$user_id
# export PIPELINE_ID=$pipeline_id
# export DATASET_VERSION_ID=$dataset_id
# export PROJECT_ID=$project_id
unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy

python3 compare.py
