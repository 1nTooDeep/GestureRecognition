from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate
import torch
import logging

def eval(model, dataLoader, DEVICE, logger):
    print(f"logger is :{logger is None}")
    model.to(DEVICE)
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, labels in dataLoader:
            input, labels = input.to(DEVICE), labels.to(DEVICE)
            output = model(input,labels)
            _, predicted = torch.max(output.logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
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
    if logger is None:
        print("\n"+tabulate(data, headers=headers, tablefmt="pretty"))
        print("\nConfusion Matrix:")
        print("\n"+tabulate(cm, headers=["True Label", "Predicted Label"], tablefmt="grid"))
    else:
        logger.info("\n"+tabulate(data, headers=headers, tablefmt="pretty"))
        logger.info("\nConfusion Matrix:")
        logger.info("\n"+tabulate(cm, headers=["True Label", "Predicted Label"], tablefmt="grid"))
    # 返回计算结果
    return accuracy, recall, precision, f1, cm