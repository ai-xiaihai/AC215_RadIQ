# use python image as base image
FROM python:3.8-slim-buster

# set up environment variable
ENV PYENV_SHELL=/bin/bash

# Install pipenv
RUN pip install pipenv

# working directory
WORKDIR /src

# Copy scripts into container
COPY data_splitting.py /src/
COPY Pipfile /src/

# Install dependencies for data splitting
RUN pipenv install --deploy --ignore-pipfile

# Execute the scripts
CMD ["pipenv", "run", "python", "data_splitting.py"]