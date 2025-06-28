import os
import time
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import environment Anda yang sudah di-upgrade (Fase 2)
# Pastikan nama file ini sudah benar
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
    # Anda bisa mengatur ulang total_timesteps untuk sesi fine-tuning ini
    # Misalnya, Anda ingin menambahkan 2 juta step lagi
    total_timesteps = int(2e6) 
    log_dir = "./sb3_logs_rovid_phase2/" # Ganti nama log dir agar tidak tercampur
    os.makedirs(log_dir, exist_ok=True)

    print(f"Menyiapkan {num_cpu} environment training paralel...")
    
    # 1. Buat Vectorized Environment untuk Training (menggunakan environment Fase 2)
    train_env = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=log_dir)
        for i in range(num_cpu)
    ])

    # --- PERUBAHAN DIMULAI DI SINI: MEMUAT MODEL ATAU MEMBUAT BARU ---

    # 1. Tentukan path ke model yang ingin Anda muat (model 3 juta step Anda)
    # GANTI PATH INI dengan path yang benar ke file .zip model Anda
    model_to_load_path = "/home/rafaalfe/new/DRL-robot-navigation/TD3/sb3_logs_rovid_final/best_model/best_model.zip" 

    if os.path.exists(model_to_load_path):
        print("="*50)
        print(f"MEMUAT MODEL YANG SUDAH ADA DARI: {model_to_load_path}")
        print("="*50)
        
        # Muat model yang sudah ada. Environment (train_env) harus disediakan.
        # Anda juga bisa menimpa hyperparameter di sini jika perlu,
        # misalnya: learning_rate=1e-5 untuk fine-tuning yang lebih halus.
        model = TD3.load(
            model_to_load_path,
            env=train_env,
            # Anda bisa menimpa beberapa parameter jika mau, contoh:
            learning_rate=1e-5 
        )
        print("Model berhasil dimuat.")
    else:
        print("="*50)
        print(f"PERINGATAN: File model di '{model_to_load_path}' tidak ditemukan.")
        print("Membuat model baru dari awal...")
        print("="*50)

        # Jika file tidak ditemukan, kode ini akan membuat model baru
        action_dim = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
        policy_kwargs = dict(net_arch=[1024, 512])

        model = TD3(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            buffer_size=int(2e5),
            batch_size=256,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir
        )
    # --- AKHIR DARI BLOK PERUBAHAN ---

    # --- KONFIGURASI CALLBACK ---
    # Nama prefix untuk checkpoint baru bisa diganti agar tidak menimpa yang lama
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(log_dir, "checkpoints"),
      name_prefix="rovid_finetuned_model"
    )

    eval_callback = EvalCallback(
        train_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=num_cpu * 2,
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # --- PROSES TRAINING (FINE-TUNING) ---
    print("="*50)
    print("Memulai proses training (atau fine-tuning)...")
    print("="*50)
    
    # --- PERUBAHAN KEDUA: TAMBAHKAN 'reset_num_timesteps=False' ---
    # Ini penting agar penghitungan timestep di log tidak dimulai dari 0 lagi.
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        reset_num_timesteps=False # <-- JANGAN LUPAKAN INI
    )

    # Simpan model final dari sesi fine-tuning
    model.save(os.path.join(log_dir, "rovid_model_finetuned_final"))

    train_env.close()
    print("Proses fine-tuning selesai.")