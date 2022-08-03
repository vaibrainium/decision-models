FROM python 3.9.13  # Basic python image
LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'

# Copy the code into the container and install it
COPY . /src/decision_models
RUN pip install --no-cache /src/decision_models

