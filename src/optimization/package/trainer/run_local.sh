export GOOGLE_APPLICATION_CREDENTIALS="../../../../secrets/model-trainer.json"
rm model.py
rm dataset_mscxr.py
rm -r health_multimodal
python train.py