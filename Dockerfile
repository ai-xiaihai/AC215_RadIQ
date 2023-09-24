# use python image as base image
FROM python:3.8-slim-buster

ENV PYENV_SHELL=/bin/bash

# Copy scripts into container
COPY . .

# working directory
WORKDIR /src

# Install dependencies for data preprocessing
RUN pip install pandas numpy scikit-learn

# Execute the scripts
CMD ["bash", "run.sh"]