## Docker Image Info for MLflow

Docker Hub Link:
https://hub.docker.com/r/YOUR_USERNAME/YOUR_IMAGE_NAME

To build image from an MLflow run:

```bash
mlflow models build-docker -m runs:/<run_id>/model -n diabetes_rf_image
