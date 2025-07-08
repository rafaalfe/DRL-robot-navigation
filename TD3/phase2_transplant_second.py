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

# --- KONFIGURASI ---
NEW_NUM_CPU = 10
LOG_DIR = f"./rovid_phase2_{NEW_NUM_CPU}cpu/"
TOTAL_TIMESTEPS = int(20e6) # Naikkan total timesteps untuk training lanjutan
BATCH_SIZE = 256

# --- PERUBAHAN UTAMA: Tentukan checkpoint mana yang akan dilanjutkan ---
# Ganti nama file ini dengan checkpoint terakhir/terbaik Anda.
# Contoh: Jika training berhenti di 4.7M, mungkin ada checkpoint di 4.7M atau 4.65M.
CHECKPOINT_TO_RESUME = os.path.join(LOG_DIR, "checkpoints/rovid_finetuned_model_4730000_steps.zip")


if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\nMenyiapkan {NEW_NUM_CPU} environment training paralel...")
    train_env_baru = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=LOG_DIR)
        for i in range(NEW_NUM_CPU)
    ])
    
    reset_num_timesteps = True
    
    # --- LOGIKA BARU UNTUK MELANJUTKAN TRAINING ---
    if os.path.exists(CHECKPOINT_TO_RESUME):
        print("="*60)
        print(f"DITEMUKAN CHECKPOINT. MELANJUTKAN TRAINING DARI:")
        print(f"{CHECKPOINT_TO_RESUME}")
        print("="*60)
        
        # 1. Muat model dari file checkpoint
        # Penting: Berikan environment baru (`train_env_baru`) saat memuat.
        model_baru = TD3.load(
            CHECKPOINT_TO_RESUME,
            env=train_env_baru,
            tensorboard_log=LOG_DIR
        )
        
        # 2. Muat replay buffer yang sesuai
        replay_buffer_path = "/home/rafaalfe/training_phase2/rovid_phase2_10cpu/checkpoints/rovid_finetuned_model_replay_buffer_4730000_steps.pkl"
        print(f"[*] Memuat Replay Buffer dari: {replay_buffer_path}")
        model_baru.load_replay_buffer(replay_buffer_path)
        
        # 3. Pastikan timestep tidak di-reset
        reset_num_timesteps = False

    else:
        # Fallback jika checkpoint tidak ditemukan: Latih dari nol
        print(f"\n[WARNING] File checkpoint tidak ditemukan di {CHECKPOINT_TO_RESUME}.")
        print("Memulai sesi training baru dari awal.")
        
        action_dim = train_env_baru.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

        model_baru = TD3(
            "MlpPolicy",
            train_env_baru,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[1024, 512]),
            learning_rate=1e-4,
            buffer_size=int(5e5), # Pertimbangkan untuk memperbesar buffer
            batch_size=BATCH_SIZE,
            gamma=0.99,
            learning_starts=BATCH_SIZE,
            verbose=1,
            tensorboard_log=LOG_DIR
        )

    # Konfigurasi callback tetap sama
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(LOG_DIR, "checkpoints"),
      name_prefix="rovid_finetuned_model",
      save_replay_buffer=True, # Pastikan ini tetap True
      save_vecnormalize=True
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
    
    print(f"\nMemulai sesi training lanjutan...")
    
    # Panggil .learn() dengan reset_num_timesteps yang sudah diatur
    model_baru.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=10,
        reset_num_timesteps=reset_num_timesteps
    )

    model_baru.save(os.path.join(LOG_DIR, "rovid_model_final_run"))
    train_env_baru.close()
    
    print("\nProses training telah selesai.")
