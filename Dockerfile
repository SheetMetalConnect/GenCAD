FROM continuumio/miniconda3

ENV CONDA_ENV_NAME=gencad_env
WORKDIR /app

# Install system dependencies (including xvfb and OpenGL libs)
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    mesa-utils \
    fonts-dejavu-core \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "gencad_env", "/bin/bash", "-c"]

COPY . .

EXPOSE 7860
CMD ["conda", "run", "-n", "gencad_env", "xvfb-run", "--server-args=-screen 0 2048x2048x24", "python", "demo.py"]
