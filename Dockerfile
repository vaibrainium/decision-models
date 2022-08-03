FROM cuda-11.6.1-pytorch-1.12.0-torchvision-0.12.0-torchaudio-0.11.0-ubuntu-20.04  # Walker lab pytorch image
LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'

# Copy the code into the container and install it
COPY . /src/decision_models
RUN pip install --no-cache /src/decision_models

