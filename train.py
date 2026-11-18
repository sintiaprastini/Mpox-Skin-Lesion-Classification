import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils.getData import getImageLabel, CLASSES
from Models.CNN import SimpleCNN, ModifiedCNN, ExternalDL

torch.set_num_threads(os.cpu_count() or 4)

BASE_ORIG = r"D:\Tubes_Mpox\Mpox_Dataset\Original Images\Original Images\FOLDS"
BASE_AUG  = r"D:\Tubes_Mpox\Mpox_Dataset\Augmented Images\Augmented Images\FOLDS_AUG"
os.makedirs("results", exist_ok=True)
os.makedirs("Checkpoints", exist_ok=True)


def get_model(algo_name):
    if algo_name == "SimpleCNN":
        return SimpleCNN(input_c=3, output=6)
    elif algo_name == "ModifiedCNN":
        return ModifiedCNN(input_c=3, output=6, dropout=0.5)
    elif algo_name == "MobileNetV3":
        return ExternalDL(output=6, dropout=0.5)


def compute_class_weights(dataset):
    """
    Menghitung class weight berbanding terbalik dengan frekuensi kelas
    untuk mengantisipasi data yang tidak seimbang (imbalanced dataset).
    Rumus: w_i = N_total / (N_classes * count_i)
    """
    class_indices = torch.argmax(dataset.labels, dim=1).numpy()
    counts = np.bincount(class_indices, minlength=len(CLASSES))
    total = len(class_indices)
    num_classes = len(CLASSES)

    # Gunakan smoothed inverse frequency agar bobot seimbang & stabil
    weights = total / (num_classes * np.maximum(counts, 1))
    weights = np.sqrt(weights)  # akar kuadrat agar penalti proporsional dan tidak merusak gradien
    weights = weights / np.mean(weights)

    print("   Distribusi Kelas Data Train & Class Weights:")
    for cls_name, cnt, w in zip(CLASSES, counts, weights):
        print(f"      - {cls_name:10s}: {cnt:4d} citra | weight = {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def main():
    BATCH_SIZE = 64
    EPOCH = 40
    LEARNING_RATE = 0.001
    N_FOLDS = 5
    ALGORITHMS = ["SimpleCNN", "ModifiedCNN", "MobileNetV3"]

    for fold in range(1, N_FOLDS + 1):
        print(f"\n=======================================================")
        print(f"                  MEMUAT DATA FOLD {fold} / {N_FOLDS}")
        print(f"=======================================================")
        train_orig = os.path.join(BASE_ORIG, f"fold{fold}", "Train")
        train_aug  = os.path.join(BASE_AUG,  f"fold{fold}_AUG", "Train")
        valid_dir  = os.path.join(BASE_ORIG, f"fold{fold}", "Valid")

        need_load = any(
            not os.path.exists(f"./Checkpoints/{algo}_fold{fold}.pt")
            for algo in ALGORITHMS
        )

        if not need_load:
            print(f"Semua algoritma pada Fold {fold} sudah dilatih. Lanjut ke fold berikutnya...")
            continue

        print("Memuat & pre-processing gambar ke memori...")
        only_mobilenet = (
            os.path.exists(f"./Checkpoints/SimpleCNN_fold{fold}.pt")
            and os.path.exists(f"./Checkpoints/ModifiedCNN_fold{fold}.pt")
            and not os.path.exists(f"./Checkpoints/MobileNetV3_fold{fold}.pt")
        )
        img_sz = 128 if only_mobilenet else 64
        train_data = getImageLabel(folders=[train_orig, train_aug], img_size=img_sz)
        valid_data = getImageLabel(folders=[valid_dir], img_size=img_sz)
        print(f"Jumlah Data -> Train: {len(train_data)} | Valid: {len(valid_data)} (Ukuran: {img_sz}x{img_sz})")

        # Hitung class weight untuk antisipasi imbalanced dataset
        class_weights = compute_class_weights(train_data)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

        for algo in ALGORITHMS:
            ckpt_path = f"./Checkpoints/{algo}_fold{fold}.pt"
            plot_path = f"./results/training_{algo}_fold{fold}.png"

            if os.path.exists(ckpt_path):
                print(f"\n---> [{algo}] Fold {fold} sudah selesai dilatih sebelumnya (Skip).")
                continue

            print(f"\n---> Melatih Algoritma: {algo} (Fold {fold}) <---")
            model = get_model(algo)

            # Khusus ModifiedCNN & MobileNetV3: gunakan Class Weights + L2 Regularization + LR kecil agar grafik menyatu & tidak overfitting
            if algo == "ModifiedCNN":
                print("   [Antisipasi Imbalanced Data] Menggunakan Weighted CrossEntropyLoss pada ModifiedCNN")
                optimizer = Adam(model.parameters(), lr=0.00015, weight_decay=1e-2)
                scheduler = None
                loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            elif algo == "MobileNetV3":
                print("   [Antisipasi Imbalanced & Anti-Overfitting] Menggunakan Weighted CrossEntropyLoss + AdamW + CosineAnnealingLR pada MobileNetV3")
                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=2e-2)
                scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH)
                loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            else:
                optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
                scheduler = None
                loss_function = nn.CrossEntropyLoss()

            loss_train_all, loss_valid_all = [], []
            best_val_loss = float('inf')

            for epoch in range(EPOCH):
                train_loss = 0.0
                valid_loss = 0.0

                model.train()
                for batch, (src, trg) in enumerate(train_loader):
                    if algo == "ModifiedCNN":
                        src = torch.clamp(src + 0.03 * torch.randn_like(src), 0.0, 1.0)
                    elif algo == "MobileNetV3":
                        if torch.rand(1).item() < 0.5:
                            src = torch.flip(src, dims=[3])
                        if torch.rand(1).item() < 0.5:
                            src = torch.flip(src, dims=[2])
                        k_rot = int(torch.randint(0, 4, (1,)).item())
                        if k_rot > 0:
                            src = torch.rot90(src, k_rot, dims=[2, 3])
                        src = torch.clamp(src + 0.02 * torch.randn_like(src), 0.0, 1.0)

                    pred = model(src)
                    target_idx = torch.argmax(trg, dim=1)
                    loss = loss_function(pred, target_idx)
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if scheduler is not None:
                    scheduler.step()

                model.eval()
                with torch.no_grad():
                    for batch, (src, trg) in enumerate(valid_loader):
                        pred = model(src)
                        target_idx = torch.argmax(trg, dim=1)
                        loss = loss_function(pred, target_idx)
                        valid_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)
                avg_valid_loss = valid_loss / len(valid_loader)
                loss_train_all.append(avg_train_loss)
                loss_valid_all.append(avg_valid_loss)

                print(f"[{algo} - Fold {fold}] epoch = {epoch+1:2d}/{EPOCH} | train loss = {avg_train_loss:.4f}, valid loss = {avg_valid_loss:.4f}")

                if avg_valid_loss <= best_val_loss:
                    best_val_loss = avg_valid_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_valid_loss,
                    }, ckpt_path)

            # Simpan grafik training per algoritma & fold
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, EPOCH + 1), loss_train_all, color="#931a00", label='Training Loss')
            plt.plot(range(1, EPOCH + 1), loss_valid_all, color="#3399e6", label='Validation Loss')
            plt.title(f"Training vs Validation Loss - {algo} (Fold {fold})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Grafik disimpan: {plot_path}")


if __name__ == "__main__":
    main()
