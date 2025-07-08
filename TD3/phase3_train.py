import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- Import environment baru Anda ---
# Pastikan nama file ini adalah 'phase3_env.py' atau sesuaikan
from phase3_env import MultiRoomGazeboEnv

def make_env(port, rank, log_dir):
    """
    Fungsi helper untuk membuat dan memonitor environment.
    """
    def _init():
        monitor_log_path = os.path.join(log_dir, f"monitor_{rank}")
        os.makedirs(monitor_log_path, exist_ok=True)
        env = MultiRoomGazeboEnv(port=port, launch_delay=0)
        env = Monitor(env, filename=monitor_log_path)
        return env
    return _init

# --- KONFIGURASI SESI TRAINING ---
NUM_CPU = 6
LOG_DIR = f"./rovid_phase3_path_follower/"
TOTAL_TIMESTEPS = int(10e6) # 10 Juta langkah
BUFFER_SIZE = int(5e5)
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

# --- PENTING: Atur Path Checkpoint untuk Melanjutkan ---
# Jika Anda ingin memulai dari awal, buat path ini menjadi string kosong "" atau path yang tidak ada.
# Jika ingin melanjutkan, isi dengan path ke file .zip checkpoint Anda.
# Contoh: CHECKPOINT_TO_RESUME = os.path.join(LOG_DIR, "checkpoints/path_follower_model_500000_steps.zip")
CHECKPOINT_TO_RESUME = "/home/rafaalfe/training_phase2/rovid_phase3_path_follower/checkpoints/path_follower_model_900000_steps.zip" # Biarkan kosong untuk memulai dari awal

if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)

    # Buat environment training paralel
    print(f"\nMenyiapkan {NUM_CPU} environment training paralel...")
    train_env = SubprocVecEnv([
        make_env(port=11311 + i, rank=i, log_dir=LOG_DIR)
        for i in range(NUM_CPU)
    ])
    
    # --- LOGIKA UNTUK MEMUAT ATAU MEMBUAT MODEL BARU ---
    if CHECKPOINT_TO_RESUME and os.path.exists(CHECKPOINT_TO_RESUME):
        # --- MELANJUTKAN TRAINING ---
        print("="*60)
        print(f"DITEMUKAN CHECKPOINT. MELANJUTKAN TRAINING DARI:")
        print(f"{CHECKPOINT_TO_RESUME}")
        print("="*60)
        
        # 1. Muat model dari file checkpoint
        model = TD3.load(
            CHECKPOINT_TO_RESUME,
            env=train_env,
            tensorboard_log=LOG_DIR
        )
        
        # 2. Bangun path ke replay buffer dan muat
        replay_buffer_path = "/home/rafaalfe/training_phase2/rovid_phase3_path_follower/checkpoints/path_follower_model_replay_buffer_900000_steps.pkl"
        if os.path.exists(replay_buffer_path):
            print(f"[*] Memuat Replay Buffer dari: {replay_buffer_path}")
            model.load_replay_buffer(replay_buffer_path)
        else:
            print(f"[WARNING] Replay buffer tidak ditemukan di {replay_buffer_path}. Melanjutkan dengan buffer kosong.")
        
        # Atur agar timestep di TensorBoard tidak di-reset
        reset_num_timesteps = False

    else:
        # --- MEMULAI TRAINING BARU DARI AWAL ---
        print("="*60)
        print("      MEMULAI SESI TRAINING BARU (PHASE 3 - PATH FOLLOWER)")
        print("="*60)
        print(f"Training akan dimulai dari awal (from scratch).")
        
        action_dim = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

        model = TD3(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[1024, 512]),
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=0.99,
            learning_starts=25000,
            verbose=1,
            tensorboard_log=LOG_DIR
        )
        
        # Mulai timestep dari 0
        reset_num_timesteps = True

    # Konfigurasi callback untuk menyimpan model
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path=os.path.join(LOG_DIR, "checkpoints"),
      name_prefix="path_follower_model",
      save_replay_buffer=False,
      save_vecnormalize=False
    )
    eval_callback = EvalCallback(
        train_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=100000 // NUM_CPU,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    print(f"\nMemulai sesi training...")
    if reset_num_timesteps:
        print(f"Fase 'Warm-up' akan berjalan selama {model.learning_starts} langkah pertama...")
    
    # Jalankan proses training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=10,
        reset_num_timesteps=reset_num_timesteps
    )

    model.save(os.path.join(LOG_DIR, "path_follower_model_final"))
    train_env.close()
    
    print("\nProses training telah selesai.")
