# Use an existing base image as a starting point.
FROM debian:bullseye-slim

# Install required dependencies (including curl and apt-transport-https).
RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg

# Add the Google Cloud SDK repository and GPG key.
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Google Cloud SDK and additional components.
RUN apt-get install google-cloud-sdk -y

# Set up authentication (you can configure this later).
COPY secrets/data-service-account.json /root/.config/gcloud/key.json

WORKDIR /src/data_pipeline/

# Add Pipfile.lock
COPY /src/data_pipeline/Pipfile.lock /src/data_pipeline/

RUN pip install pipenv && pipenv sync

