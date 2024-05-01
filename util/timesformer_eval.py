from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate
import torch
import logging
def top5_eval(model, device, dataloader):
    model.to(device)
    model.eval()
    acc_top_k = 0
    correct_predictions = 0
    num = 0
    k = 5
    with torch.no_grad():
        for data,labels in dataloader:
            data, labels = data.to(device),labels.to(device)
            output = model(data.to(torch.float32))
            probs = torch.softmax(output, dim=1)

            _, predicted = torch.max(output, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            _, top_k_indices = torch.topk(probs, k=5, dim=1)
            # Compare top 5 predictions with ground truth labels
            correct_top5 = (top_k_indices == labels.unsqueeze(1)).any(dim=1)

            acc_top_k += torch.sum(correct_top5)
            num += len(data)
            
    return correct_predictions / num, acc_top_k / num
def eval(model, dataLoader, DEVICE, logger):
    model.to(DEVICE)
    model.eval()
    total = 0
    correct = 0
    correct_5 = 0 
    k = 5
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input, labels in dataLoader:
            input, labels = input.to(DEVICE), labels.to(DEVICE)
            output = model(input,labels)

            prob = torch.softmax(output.logits,dim=1)
            _, predicted = torch.max(prob, 1)

            _, top_k_indices = torch.topk(prob, k=k, dim=1)

            correct_top5 = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
            correct_5 += torch.sum(correct_top5)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    accuracy_1 = correct / total
    accuracy_5 = correct_5 / total
    recall = recall_score(y_true, y_pred,average="macro")
    precision = precision_score(y_true, y_pred,average="macro")
    f1 = f1_score(y_true, y_pred,average="macro")


    # 使用 tabulate 输出指标
    headers = ["Metric", "Value"]
    data = [
        ("Top1 Accuracy", accuracy_1),
        ("Top5 Accuracy", accuracy_5),
        ("Recall", recall),
        ("Precision", precision),
        ("F1 Score", f1),
    ]
    if logger is None:
        print("\n"+tabulate(data, headers=headers, tablefmt="pretty"))
    else:
        logger.info("\n"+tabulate(data, headers=headers, tablefmt="pretty"))
    # 返回计算结果
    return accuracy_1, accuracy_5, recall, precision, f1