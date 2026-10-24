FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz python3-sklearn python3-joblib python3-notebook \
    python3-pil python3-matplotlib xvfb x11vnc novnc websockify \
    ffmpeg fonts-noto-cjk mesa-utils x11-utils openbox \
    && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99 QT_X11_NO_MITSHM=1 LIBGL_ALWAYS_SOFTWARE=1
WORKDIR /work
CMD ["bash", "scripts/start.sh"]
