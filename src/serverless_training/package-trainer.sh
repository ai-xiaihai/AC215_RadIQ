rm -f trainer.tar trainer.tar.gz 
rm -rf package/trainer/wandb
tar cvf trainer.tar package
gzip trainer.tar
gsutil cp trainer.tar.gz $GCS_BUCKET_URI/biovil-trainer.tar.gz
# push this copy onto a different bucket for the ml workflow aka pipeline job
gsutil cp trainer.tar.gz gs://xray-ml-workflow/biovil-trainer.tar.gz