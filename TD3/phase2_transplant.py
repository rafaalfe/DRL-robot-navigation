import os
import time
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from phase2_env_fusion_col_rand_start import GazeboEnv

def make_env(port, rank, log_dir):
    def _init():
        monitor_log_path = os.path.join(log_dir, f"monitor_{rank}")
        os.makedirs(monitor_log_path, exist_ok=True)
        launch_delay = 0
        env = GazeboEnv(port=port, launch_delay=launch_delay)
        env = Monitor(env, filename=monitor_log_path)
        return env
    return _init

# --- KONFIGURASI WAJIB ---
NEW_NUM_CPU = 10
MODEL_LAMA_PATH = "/home/rafaalfe/training_phase2/rovid_phase2_10cpu/best_model/best_model.zip"
LOG_DIR = f"./rovid_phase2_second_{NEW_NUM_CPU}cpu/"
BUFFER_SIZE = int(2e5) # Cukup definisikan sekali
TOTAL_TIMESTEPS = int(20e6)
BATCH_SIZE = 256

if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)

    # =========================================================================
    # --- PERUBAHAN LOGIKA DIMULAI DI SINI ---
    # =========================================================================
    # Kita tentukan nilai untuk `learning_starts` dan `reset_timesteps` terlebih dahulu
    # berdasarkan keberadaan model donor.

    reset_timesteps = True
    learning_starts_steps = BATCH_SIZE  # Default warm-up untuk buffer kosong


    if os.path.exists(MODEL_LAMA_PATH):
        print(f"[*] Ditemukan model donor di: {MODEL_LAMA_PATH}")
        # Karena kita tahu buffer donor kosong, kita tidak perlu memuatnya untuk dicek.
        # Kita hanya perlu tahu bahwa kita akan melakukan transplantasi.
        reset_timesteps = False
        # Karena bobotnya sudah pintar, kita tidak perlu warm-up.
        # Proses learning bisa langsung dimulai.
        learning_starts_steps = 0
    else:
        print(f"\n[WARNING] File model donor tidak ditemukan. Training dari nol.")
    # =========================================================================

    print(f"\nMenyiapkan {NEW_NUM_CPU} environment training paralel...")
    train_env_baru = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=LOG_DIR)
        for i in range(NEW_NUM_CPU)
    ])
    action_dim = train_env_baru.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.02 * np.ones(action_dim))


    print(f"Menciptakan struktur model baru untuk {NEW_NUM_CPU} CPU...")
    # Masukkan `learning_starts` saat INISIALISASI model
    model_baru = TD3(
        "MlpPolicy",
        train_env_baru,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[1024, 512]),
        learning_rate=1e-5,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=0.99,
        learning_starts=learning_starts_steps, # <-- PARAMETER DITEMPATKAN DI SINI
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    # Lakukan transplantasi JIKA model lama ada
    if not reset_timesteps: # `reset_timesteps` adalah False jika model lama ada
        print("="*60)
        print("                 MEMULAI PROSEDUR TRANSPLANTASI STATE")
        print("="*60)
        
        model_lama = TD3.load(MODEL_LAMA_PATH, device='cpu')
        
        print("[*] Mentransfer Bobot Jaringan Saraf...")
        model_baru.policy.load_state_dict(model_lama.policy.state_dict())
        
        print("[*] Mentransfer State Optimizer...")
        model_baru.actor.optimizer.load_state_dict(model_lama.actor.optimizer.state_dict())
        model_baru.critic.optimizer.load_state_dict(model_lama.critic.optimizer.state_dict())
        
        print("[*] Sinkronisasi Progres Training...")
        model_baru.num_timesteps = model_lama.num_timesteps
        model_baru._episode_num = model_lama._episode_num
        model_baru._total_timesteps = model_lama._total_timesteps
        
        del model_lama
        print("\n[SUCCESS] Transplantasi Selesai!")
        print("="*60)

    # Konfigurasi callback
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(LOG_DIR, "checkpoints"),
      name_prefix="rovid_finetuned_model",
      save_replay_buffer=False,
      save_vecnormalize=False
    )
    eval_callback = EvalCallback(
        train_env_baru,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=100000 // NEW_NUM_CPU,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    print(f"\nMemulai sesi training/fine-tuning pada {NEW_NUM_CPU} environment...")
    if learning_starts_steps > 0:
        print(f"Fase 'Warm-up' (learning_starts) akan berjalan selama {learning_starts_steps} langkah pertama...")

    # Panggil .learn() TANPA 'learning_starts'
    model_baru.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=10,
        reset_num_timesteps=reset_timesteps
    )

    model_baru.save(os.path.join(LOG_DIR, "rovid_model_final_run"))
    train_env_baru.close()
    
    print("\nProses training/fine-tuning telah selesai.")