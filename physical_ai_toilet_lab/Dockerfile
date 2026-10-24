FROM physical-ai-step1:jazzy
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-deps python-fcl==0.7.0.8
WORKDIR /work/scripts
