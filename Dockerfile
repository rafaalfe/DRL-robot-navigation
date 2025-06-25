# Tahap 1: Memilih Fondasi
# Kita mulai dari image resmi ROS Noetic, yang sudah berisi Ubuntu 20.04 dan ROS.
FROM ros:noetic

# Setel shell default ke bash untuk konsistensi
SHELL ["/bin/bash", "-c"]


# Tahap 2: Mengatur Environment & Menginstal Dependensi Sistem
# DEBIAN_FRONTEND=noninteractive mencegah 'apt-get' berhenti dan meminta input dari user.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    nano \
    tmux \
    ros-noetic-xacro \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-rviz \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    tmux \
    && rm -rf /var/lib/apt/lists/*


# Tahap 3: Menginstal Library Python
# Kita gabungkan dalam satu layer untuk sedikit optimasi dan install semua dependensi Python.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    stable-baselines3[extra] \
    squaternion \
    rospkg


# Tahap 4: Menyiapkan dan Mem-build Workspace ROS
# Buat direktori untuk catkin workspace di dalam image
WORKDIR /root/catkin_ws

# Salin HANYA folder 'src' dari catkin_ws Anda ke dalam image
# Asumsi Dockerfile ini berada di DRL-robot-navigation/
COPY ./catkin_ws/src ./src

# Jalankan catkin_make di dalam image.
# 'source' diperlukan agar 'catkin_make' bisa ditemukan.
RUN source /opt/ros/noetic/setup.bash && catkin_make

# ---> PENYESUAIAN PENTING <---
# Tambahkan sourcing otomatis ke .bashrc untuk sesi interaktif (docker exec)
# Ini membuat Anda tidak perlu 'source' manual setiap kali membuka terminal baru.
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc


# Tahap 5: Menyiapkan Direktori Training DRL
# Buat direktori terpisah untuk script training agar rapi
WORKDIR /root/drl_ws

# Salin folder TD3 Anda (yang berisi rovid_env.py, train_parallel.py, dll.)
# Asumsi Dockerfile ini berada di DRL-robot-navigation/
COPY ./TD3 .


# Tahap 6: Menyiapkan Entrypoint
# Entrypoint adalah script yang akan selalu dijalankan pertama kali saat kontainer dimulai
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Perintah default jika kontainer dijalankan tanpa perintah lain
CMD ["bash"]