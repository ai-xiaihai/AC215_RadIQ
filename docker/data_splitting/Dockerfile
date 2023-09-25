# use python image as base image
FROM python:3.8-slim-buster

ENV PYENV_SHELL=/bin/bash

# working directory
WORKDIR /src

# Copy scripts into container
COPY ../../src/ /src

# Install dependencies for data preprocessing
RUN pip install pandas numpy scikit-learn

# Execute the scripts
CMD ["python", "data_splitting.py"]