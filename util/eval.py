from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate
import torch
import logging

def eval(model, dataLoader, DEVICE, logger):
    model.to(DEVICE)
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for val_data, val_target in dataLoader:
            val_data, val_target = val_data.to(DEVICE), val_target.to(DEVICE)
            output = model(val_data)
            _, predicted = torch.max(output.data, 1)

            total += val_target.size(0)
            correct += (predicted == val_target).sum().item()
            y_true.extend(val_target.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = correct / total

    recall = recall_score(y_true, y_pred,average="macro")
    precision = precision_score(y_true, y_pred,average="macro")
    f1 = f1_score(y_true, y_pred,average="macro")

    cm = confusion_matrix(y_true, y_pred)

    # 使用 tabulate 输出指标
    headers = ["Metric", "Value"]
    data = [
        ("Accuracy", accuracy),
        ("Recall", recall),
        ("Precision", precision),
        ("F1 Score", f1),
    ]
    print(tabulate(data, headers=headers, tablefmt="pretty"))
    logger.info(tabulate(data, headers=headers, tablefmt="pretty"))
    print("\nConfusion Matrix:")
    logger.info("\nConfusion Matrix:")
    print(tabulate(cm, headers=["True Label", "Predicted Label"], tablefmt="grid"))
    logger.info(tabulate(cm, headers=["True Label", "Predicted Label"], tablefmt="grid"))
    # 返回计算结果
    return accuracy, recall, precision, f1, cm