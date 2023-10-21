from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "wandb==0.15.11", 
    "huggingface-hub==0.6.0",
    "pydicom==2.2.2",
    "SimpleITK==2.1.1.1",
    "timm==0.6.5",
    "torchvision==0.14.1",
    "transformers==4.17.0",
    "scikit-image==0.22.0",
    "protobuf==4.24.4",
    ]

setup(
    name="biovil-app-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="BioViL App Trainer Application",
)
