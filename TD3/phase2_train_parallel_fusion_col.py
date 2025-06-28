import os
import time
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import environment Anda
from phase2_env_fusion_col_rand_start import GazeboEnv

# Fungsi 'maker' tetap sama
def make_env(port, rank, log_dir):
    def _init():
        monitor_log_path = os.path.join(log_dir, f"monitor_{rank}")
        os.makedirs(monitor_log_path, exist_ok=True)
        
        launch_delay = rank * 20
        env = GazeboEnv(port=port, launch_delay=launch_delay)
        
        env = Monitor(env, filename=monitor_log_path)
        return env
    return _init

if __name__ == '__main__':
    # --- KONFIGURASI UTAMA ---
    num_cpu = 3
    total_timesteps = int(20e6)
    log_dir = "./sb3_logs_rovid_final/" # Ganti nama log dir agar tidak tercampur
    os.makedirs(log_dir, exist_ok=True)

    print(f"Menyiapkan {num_cpu} environment training paralel...")
    
    # 1. Buat Vectorized Environment untuk Training
    train_env = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=log_dir)
        for i in range(num_cpu)
    ])

    # --- PERUBAHAN: Hapus pembuatan eval_env yang terpisah ---
    # print("Menyiapkan environment evaluasi...")
    # eval_env = make_env(port=11399, rank=99, log_dir=log_dir)()

    # --- KONFIGURASI MODEL DRL (TD3) ---
    action_dim = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
    policy_kwargs = dict(net_arch=[1024, 512])

    model = TD3(
        "MlpPolicy",
        train_env, # Train pada environment paralel
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=int(2e5),
        batch_size=256,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir
    )

    # --- KONFIGURASI CALLBACK ---

    # Callback #1: Checkpoint untuk backup
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(log_dir, "checkpoints"),
      name_prefix="rovid_model"
    )

    # Callback #2: Evaluasi
    # --- PERUBAHAN: Gunakan 'train_env' sebagai 'eval_env' ---
    eval_callback = EvalCallback(
        train_env, # <- Gunakan environment training untuk evaluasi
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=num_cpu * 2, # Lakukan evaluasi 2 episode per environment
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # --- PROSES TRAINING ---
    print("Memulai proses training dengan evaluasi di environment yang sama...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10
    )

    model.save(log_dir + "rovid_model_final")

    # --- PERUBAHAN: Hapus penutupan eval_env yang sudah tidak ada ---
    train_env.close()
    # eval_env.close() 
    print("Training selesai.")
