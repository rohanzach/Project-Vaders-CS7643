# Project-Vaders: Qwen3-TTS Research Lab
This repository contains the source code for the Project-Vaders voice cloning system, built on the Qwen3-TTS architecture. This setup is optimized for NVIDIA GPU acceleration and includes pre-compiled support for Flash-Attention 2.

## 🚀 Getting Started
Follow these steps to clone the repository and build the research environment.

1. Clone the Repository
First, clone the project repository to your local machine:

Bash
git clone https://github.com/rohanzach/Project-Vaders-CS7643.git
cd Project-Vaders-CS7643
2. Build the Docker Image
The build process handles the installation of CUDA 12.4, Python 3.10, and the compilation of Flash-Attention.

Bash
docker build -t vader_env .
Note: The flash-attn compilation step can take 10–15 minutes depending on your CPU/GPU. Ensure you have psutil and ninja installed in your Dockerfile (as configured) to prevent build timeouts.

## 🛠️ Launching the Environment
To begin research or run inference, launch the container in Interactive Mode. This command mounts your current directory into the container, allowing for real-time code edits on your host machine (via VS Code) that persist inside the environment.

Launch Command
Bash
sudo docker run -d -it -v "$(pwd):/workspace" --gpus all vader_env

## 🧪 Verifying the Installation
Try running the 1st cell in test.ipynb
