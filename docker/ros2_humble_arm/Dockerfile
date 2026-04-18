FROM ubuntu:22.04

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG HOST_DIALOUT_GID=
ARG HOST_VIDEO_GID=
ARG HOST_RENDER_GID=
ARG HOST_PLUGDEV_GID=
ARG HOST_I2C_GID=
ARG HOST_GPIO_GID=
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV WORKSPACE_IMAGE_PATH=/opt/Engineer2026_Control
ENV WORKSPACE_HOST_PATH=/workspace/Engineer2026_Control_host
ENV WORKSPACE_ACTIVE_PATH=/workspace/Engineer2026_Control

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    ca-certificates && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    add-apt-repository universe && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME}") main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    bash-completion \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-serial \
    python3-numpy \
    python3-scipy \
    python3-yaml \
    python3-pykdl \
    x11-apps \
    mesa-utils \
    usbutils \
    v4l-utils \
    ros-humble-desktop \
    ros-humble-moveit \
    ros-humble-moveit-servo \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-xacro && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init || true

RUN set -eux; \
    ensure_group() { \
      name="$1"; \
      gid="$2"; \
      if [ -z "${gid}" ]; then \
        return 0; \
      fi; \
      if getent group "${name}" >/dev/null; then \
        return 0; \
      fi; \
      if getent group "${gid}" >/dev/null; then \
        return 0; \
      fi; \
      groupadd --gid "${gid}" "${name}"; \
    }; \
    ensure_group dialout "${HOST_DIALOUT_GID}"; \
    ensure_group video "${HOST_VIDEO_GID}"; \
    ensure_group render "${HOST_RENDER_GID}"; \
    ensure_group plugdev "${HOST_PLUGDEV_GID}"; \
    ensure_group i2c "${HOST_I2C_GID}"; \
    ensure_group gpio "${HOST_GPIO_GID}"

RUN groupadd --gid "${GID}" "${USERNAME}" && \
    useradd --uid "${UID}" --gid "${GID}" -m -s /bin/bash "${USERNAME}" && \
    usermod -aG dialout,video,plugdev,render,i2c,gpio "${USERNAME}" || true

COPY docker/ros2_humble_arm/ros_entrypoint.sh /ros_entrypoint.sh
COPY . /opt/Engineer2026_Control
RUN chmod +x /ros_entrypoint.sh

RUN mkdir -p /workspace/Engineer2026_Control_host && \
    ln -s "${WORKSPACE_IMAGE_PATH}" "${WORKSPACE_ACTIVE_PATH}" && \
    chown -R "${UID}:${GID}" /workspace "${WORKSPACE_IMAGE_PATH}" /home/"${USERNAME}"

USER ${USERNAME}
WORKDIR /workspace/Engineer2026_Control

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
