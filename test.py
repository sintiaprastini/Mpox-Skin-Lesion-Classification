import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from Utils.getData import getImageLabel, CLASSES
from Models.CNN import SimpleCNN, ModifiedCNN, ExternalDL
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)

BASE_ORIG = r"D:\Tubes_Mpox\Mpox_Dataset\Original Images\Original Images\FOLDS"
os.makedirs("results", exist_ok=True)


def get_model(algo_name):
    if algo_name == "SimpleCNN":
        return SimpleCNN(input_c=3, output=6)
    elif algo_name == "ModifiedCNN":
        return ModifiedCNN(input_c=3, output=6, dropout=0.5)
    elif algo_name == "MobileNetV3":
        return ExternalDL(output=6, dropout=0.5)


def main():
    BATCH_SIZE = 64
    N_FOLDS = 5
    ALGORITHMS = ["SimpleCNN", "ModifiedCNN", "MobileNetV3"]

    # Load kelima fold test set sesuai ukuran input masing-masing model
    test_loaders_64 = {}
    test_loaders_128 = {}
    for fold in range(1, N_FOLDS + 1):
        test_dir = os.path.join(BASE_ORIG, f"fold{fold}", "Test")
        test_loaders_64[fold] = DataLoader(getImageLabel(folders=[test_dir], img_size=64), batch_size=BATCH_SIZE, shuffle=False)
        test_loaders_128[fold] = DataLoader(getImageLabel(folders=[test_dir], img_size=128), batch_size=BATCH_SIZE, shuffle=False)

    summary_comparison = {}

    for algo in ALGORITHMS:
        print(f"\n=======================================================")
        print(f"  EVALUASI ALGORITMA: {algo}")
        print(f"=======================================================")

        acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
        cm_total = np.zeros((len(CLASSES), len(CLASSES)))
        active_loaders = test_loaders_128 if algo == "MobileNetV3" else test_loaders_64

        for fold in range(1, N_FOLDS + 1):
            ckpt_path = f'./Checkpoints/{algo}_fold{fold}.pt'
            model = get_model(algo)
            checkpoint = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            prediction, ground_truth, probabilities = [], [], []
            with torch.no_grad():
                model.eval()
                for batch, (src, trg) in enumerate(active_loaders[fold]):
                    logits = model(src)
                    probs = torch.softmax(logits, dim=1)

                    prediction.extend(torch.argmax(probs, dim=1).numpy())
                    ground_truth.extend(torch.argmax(trg, dim=1).numpy())
                    probabilities.extend(probs.numpy())

            y_true = np.array(ground_truth)
            y_pred = np.array(prediction)
            y_prob = np.array(probabilities)
            y_onehot = np.eye(len(CLASSES))[y_true]

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            roc  = roc_auc_score(y_onehot, y_prob, multi_class='ovr', average='weighted')

            acc_list.append(acc); prec_list.append(prec); rec_list.append(rec)
            f1_list.append(f1);   auc_list.append(roc)

            print(f"[{algo} - Fold {fold}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

            cf_matrix = confusion_matrix(y_true, y_pred)
            cm_total += cf_matrix

            # Simpan ROC Curve per fold untuk setiap algoritma
            plt.figure(figsize=(9, 6))
            for i, cls in enumerate(CLASSES):
                fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"ROC Curve - {algo} (Fold {fold})")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
            plt.savefig(f'./results/roc_{algo}_fold{fold}.png', dpi=150)
            plt.close()

        # Simpan Rata-rata Confusion Matrix 5-Fold per algoritma
        df_cm_avg = pd.DataFrame(cm_total / N_FOLDS, index=CLASSES, columns=CLASSES)
        plt.figure(figsize=(9, 7))
        sn.heatmap(df_cm_avg, annot=True, fmt=".1f", cmap="Blues")
        plt.title(f"Averaged Confusion Matrix (5-Fold) - {algo}")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f'./results/confusion_matrix_{algo}.png', dpi=150)
        plt.close()

        summary_comparison[algo] = {
            "Accuracy":  (np.mean(acc_list),  np.std(acc_list)),
            "Precision": (np.mean(prec_list), np.std(prec_list)),
            "Recall":    (np.mean(rec_list),  np.std(rec_list)),
            "F1-Score":  (np.mean(f1_list),   np.std(f1_list)),
            "ROC-AUC":   (np.mean(auc_list),  np.std(auc_list)),
        }

        print(f"\n>>> RERATA 5-FOLD [{algo}] <<<")
        for k, (m, s) in summary_comparison[algo].items():
            print(f"Mean {k:9s} = {m:.4f} (± {s:.4f})")

    # Tampilkan Tabel Perbandingan Akhir 3 Algoritma
    print("\n=====================================================================")
    print("        PERBANDINGAN RERATA 5-FOLD DARI KETIGA ALGORITMA")
    print("=====================================================================")
    print(f"{'Algoritma':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 77)
    best_algo = None
    best_acc = -1.0
    for algo, metrics in summary_comparison.items():
        acc_m  = metrics["Accuracy"][0]
        prec_m = metrics["Precision"][0]
        rec_m  = metrics["Recall"][0]
        f1_m   = metrics["F1-Score"][0]
        auc_m  = metrics["ROC-AUC"][0]
        print(f"{algo:<15} | {acc_m:<10.4f} | {prec_m:<10.4f} | {rec_m:<10.4f} | {f1_m:<10.4f} | {auc_m:<10.4f}")
        if acc_m > best_acc:
            best_acc = acc_m
            best_algo = algo
    print("=====================================================================")
    print(f"Kesimpulan: Algoritma paling akurat adalah [{best_algo}] dengan Mean Accuracy = {best_acc:.4f}")


if __name__ == "__main__":
    main()
