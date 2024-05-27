from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate
import torch
import logging


def eval(model, dataLoader, DEVICE, logger):
    model.to(DEVICE)
    model.eval()
    total = 0
    correct_1 = 0
    correct_5 = 0
    k = 5
    total_loss = 0
    i = 0
    y_true = []
    y_pred = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for input, labels in dataLoader:
            input, labels = input.to(DEVICE), labels.to(DEVICE)
            output = model(input)
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            i += 1
            _, predicted = torch.max(output, 1)
            prob = torch.softmax(output, dim=1)

            _, top_k_indices = torch.topk(prob, k=k, dim=1)

            correct_top5 = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
            correct_5 += torch.sum(correct_top5)
            total += labels.size(0)
            correct_1 += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    accuracy_1 = correct_1 / total
    accuracy_5 = correct_5 / total

    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)

    # 使用 tabulate 输出指标
    headers = ["Metric", "Value"]
    data = [
        ("Top1 Accuracy", accuracy_1),
        ("Top5 Accuracy", accuracy_5),
        ("Loss", total_loss / i),
        ("Recall", recall),
        ("Precision", precision),
        ("F1 Score", f1),
    ]
    logger.info("\n" + tabulate(data, headers=headers, tablefmt="pretty"))
    # 返回计算结果
    return accuracy_1, accuracy_5, total_loss / i, recall, precision, f1
