import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
from Dataset import MyDataSet
from Model import MyModel
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--best_model_path", type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = pd.read_csv("/mnt/data/lly/LW/multi_se/data_with_shape/human/human_test_with_shape.bed", sep="\t", low_memory=False)
X_test = test_data["sequence"].tolist()
MGW_test = test_data["MGW"].tolist()
EP_test = test_data["EP"].tolist()
ProT_test = test_data["ProT"].tolist()
Roll_test = test_data["Roll"].tolist()
HelT_test = test_data["HelT"].tolist()
y_test = test_data["label"].tolist()

test_dataset = MyDataSet(X_test, MGW_test, EP_test, ProT_test, Roll_test, HelT_test, y_test)

test_dataloader = Data.DataLoader(test_dataset, batch_size=16, shuffle=False)

best_model_path = args.best_model_path

loss_fn = nn.CrossEntropyLoss()

# 训练结束后，加载最佳模型并测试其性能
best_model = MyModel().to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

# 在测试集上测试最好的模型
all_test_preds = []
all_test_probs = []
all_test_targets = []
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs1, a, b, inputs2, c, d, inputs3, target in tqdm(test_dataloader, desc='Testing Best Model', leave=False):
        inputs1, a, b, inputs2, c, d, inputs3, target = inputs1.to(device), a.to(device), b.to(device), inputs2.to(device), c.to(device), d.to(device), inputs3.to(device), target.to(device)
        b = b[0]
        output, g_proj, b_proj, s_proj = best_model(inputs1, a, b, inputs2, c, d, inputs3)
        loss = loss_fn(output, target)

        test_loss += loss.item()
        _, test_predicted = torch.max(output, 1)
        test_total += target.size(0)
        test_correct += (test_predicted == target).sum().item()

        all_test_preds.extend(test_predicted.cpu().numpy())
        all_test_targets.extend(target.cpu().numpy())

        test_probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # 获取正类的概率
        all_test_probs.extend(test_probs)

avg_test_loss = test_loss / len(test_dataloader)
test_accuracy = 100 * test_correct / test_total
test_mcc = matthews_corrcoef(all_test_targets, all_test_preds)
test_auroc = roc_auc_score(all_test_targets, all_test_probs)
test_auprc = average_precision_score(all_test_targets, all_test_probs)
test_tn, test_fp, test_fn, test_tp = confusion_matrix(all_test_targets, all_test_preds).ravel()
test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) != 0 else 0
test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) != 0 else 0

print('--------------------------------------------------------')
print(f'Best Model Performance on Test Set:')
print(f'Test Loss: {avg_test_loss: .4f}')
print(f'Test Accuracy: {test_accuracy: .4f}%, Test MCC: {test_mcc: .4f}')
print(f'Test SN: {test_sensitivity: .4f}, Test SP: {test_specificity: .4f}')
print(f'Test AUROC: {test_auroc: .4f}, Test AUPRC: {test_auprc: .4f}')
