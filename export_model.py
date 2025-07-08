import torch
from stable_baselines3 import TD3

# Path ke model terlatih Anda
MODEL_PATH = "/home/rafaalfe/training_phase2/rovid_phase2_10cpu/best_model/best_model.zip"
# Path untuk menyimpan model yang sudah diekspor
EXPORT_PATH = "drl_actor_model.pt"

# --- PERBAIKAN DI SINI ---
# 1. Tentukan perangkat secara otomatis (GPU jika tersedia, jika tidak CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Muat model dan pastikan berada di perangkat yang benar
model = TD3.load(MODEL_PATH, device=device)

# Ekstrak policy 'actor' (bagian yang memilih aksi)
actor_model = model.policy.actor
actor_model.eval() # Set ke mode evaluasi

# 2. Buat contoh input DI PERANGKAT YANG SAMA dengan model
example_input = torch.randn(1, 1294, device=device) 

# Lacak model dan simpan sebagai TorchScript
print("Tracing model...")
traced_script_module = torch.jit.trace(actor_model, example_input)
traced_script_module.save(EXPORT_PATH)

print(f"Model berhasil diekspor ke: {EXPORT_PATH}")