import os
import time
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ==============================================================================
# PASTIKAN NAMA FILE ENVIRONMENT ANDA SUDAH BENAR
# Ini adalah environment Fase 2 Anda dengan start acak, collision baru, 
# dan kecepatan mundur.
from phase2_env_fusion_col_rand_start import GazeboEnv
# ==============================================================================


# Fungsi 'maker' untuk membuat environment, tidak perlu diubah.
def make_env(port, rank, log_dir):
    def _init():
        monitor_log_path = os.path.join(log_dir, f"monitor_{rank}")
        os.makedirs(monitor_log_path, exist_ok=True)
        
        launch_delay = 0
        env = GazeboEnv(port=port, launch_delay=launch_delay)
        
        env = Monitor(env, filename=monitor_log_path)
        return env
    return _init


# ==============================================================================
# --- KONFIGURASI WAJIB (HARAP DISESUAIKAN) ---
# ==============================================================================
# 1. Tentukan jumlah CPU BARU yang ingin Anda gunakan untuk fine-tuning.
NEW_NUM_CPU = 7

# 2. Tentukan path LENGKAP ke file model LAMA (.zip) yang akan menjadi "donor".
#    Contoh: "/home/kianardhani/DRL-robot-navigation/TD3/sb3_logs_rovid_final/best_model.zip"
MODEL_LAMA_PATH = "/home/kianardhani/best_model.zip"

# 3. Tentukan nama direktori log BARU untuk menyimpan hasil fine-tuning.
LOG_DIR = f"./sb3_logs_rovid_transplant_{NEW_NUM_CPU}cpu/"

# 4. Tentukan ukuran buffer dari model LAMA Anda.
#    Ini harus sama dengan yang Anda gunakan saat training Fase 1.
BUFFER_SIZE_LAMA = int(2e5)

# 5. Tentukan jumlah total timestep untuk sesi fine-tuning ini.
TOTAL_TIMESTEPS = int(2e6) # Contoh: 2 juta step tambahan
# ==============================================================================


if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- LANGKAH 1: Persiapan "Tubuh Baru" ---
    # Membuat environment baru dengan jumlah CPU yang sudah ditingkatkan.
    print(f"Mempersiapkan 'Tubuh Baru': {NEW_NUM_CPU} environment training paralel...")
    train_env_baru = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=LOG_DIR)
        for i in range(NEW_NUM_CPU)
    ])

    # --- LANGKAH 2: Membuat "Otak Kosong" ---
    # Membuat objek model TD3 baru dengan arsitektur yang benar untuk `NEW_NUM_CPU`.
    # Model ini masih "bodoh" karena bobotnya acak.
    print(f"Menciptakan 'Otak Kosong' dengan arsitektur untuk {NEW_NUM_CPU} CPU...")
    model_baru = TD3(
        "MlpPolicy",
        train_env_baru,
        policy_kwargs=dict(net_arch=[1024, 512]),
        learning_rate=1e-5,  # Menggunakan learning rate rendah untuk fine-tuning
        buffer_size=BUFFER_SIZE_LAMA,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    # --- LANGKAH 3: PROSEDUR TRANSPLANTASI ---
    # Kita cek apakah model "donor" ada. Jika ya, kita mulai operasinya.
    if os.path.exists(MODEL_LAMA_PATH):
        print("="*60)
        print("                 MEMULAI PROSEDUR TRANSPLANTASI STATE")
        print("="*60)
        
        print(f"[*] Memuat model 'donor' dari: {MODEL_LAMA_PATH}...")
        model_lama = TD3.load(MODEL_LAMA_PATH)
        
        # 3a. Transplantasi Memori Jangka Panjang (Replay Buffer)
        print("[*] Mentransfer Replay Buffer...")
        model_baru.replay_buffer.observations = model_lama.replay_buffer.observations
        model_baru.replay_buffer.actions = model_lama.replay_buffer.actions
        model_baru.replay_buffer.rewards = model_lama.replay_buffer.rewards
        model_baru.replay_buffer.next_observations = model_lama.replay_buffer.next_observations
        model_baru.replay_buffer.dones = model_lama.replay_buffer.dones
        model_baru.replay_buffer.pos = model_lama.replay_buffer.pos
        model_baru.replay_buffer.full = model_lama.replay_buffer.full
        
        # 3b. Transplantasi Otak (Bobot Network: Policy, Actor, Critic)
        print("[*] Mentransfer Bobot Jaringan Saraf...")
        model_baru.policy.load_state_dict(model_lama.policy.state_dict())
        
        # 3c. Transplantasi Momentum Belajar (State Optimizer)
        print("[*] Mentransfer State Optimizer...")
        model_baru.actor.optimizer.load_state_dict(model_lama.actor.optimizer.state_dict())
        model_baru.critic.optimizer.load_state_dict(model_lama.critic.optimizer.state_dict())
        
        # 3d. Transplantasi Catatan Riwayat (Progress Training)
        print("[*] Sinkronisasi Progres Training...")
        model_baru.num_timesteps = model_lama.num_timesteps
        model_baru._episode_num = model_lama._episode_num
        model_baru._total_timesteps = model_lama._total_timesteps
        
        # Hapus model lama dari memori untuk efisiensi
        del model_lama
        
        print("\n[SUCCESS] Prosedur Transplantasi Selesai!")
        print("="*60)
        
        reset_timesteps = False # Lanjutkan progres, jangan reset dari 0
    else:
        print(f"\n[WARNING] File model donor di '{MODEL_LAMA_PATH}' tidak ditemukan.")
        print("Training akan dimulai dari nol (tanpa transplantasi).")
        reset_timesteps = True # Mulai dari 0 karena tidak ada yang ditransfer

    # --- LANGKAH 4: Melanjutkan Training dengan Model Baru ---
    # Konfigurasi callback seperti biasa
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(LOG_DIR, "checkpoints"),
      name_prefix="rovid_transplanted_model"
    )
    eval_callback = EvalCallback(
        train_env_baru,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=NEW_NUM_CPU * 2,
        deterministic=True,
        render=False
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    print(f"\nMemulai sesi fine-tuning pada {NEW_NUM_CPU} environment...")
    
    # Gunakan 'model_baru' untuk training
    model_baru.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=10,
        reset_num_timesteps=reset_timesteps
    )

    model_baru.save(os.path.join(LOG_DIR, "rovid_model_transplanted_final"))
    train_env_baru.close()
    
    print("\nProses fine-tuning dengan transplantasi state telah selesai.")