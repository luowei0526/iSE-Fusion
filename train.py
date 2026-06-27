import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
from Dataset import MyDataSet
from Model import MyModel
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import time
import sys
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--best_model_path", type=str, default="/mnt/data/lly/LW/multi_se/best_full_fusion_model_human.pth")
parser.add_argument("--use_amp", type=lambda x: x.lower() == "true", default=True)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("/mnt/data/lly/LW/multi_se/data_with_shape/human/human_train_with_shape.bed", sep="\t", low_memory=False)
X_train = train_data["sequence"].tolist()
MGW_train = train_data["MGW"].tolist()
EP_train = train_data["EP"].tolist()
ProT_train = train_data["ProT"].tolist()
Roll_train = train_data["Roll"].tolist()
HelT_train = train_data["HelT"].tolist()
y_train = train_data["label"].tolist()

val_data = pd.read_csv("/mnt/data/lly/LW/multi_se/data_with_shape/human/human_val_with_shape.bed", sep="\t", low_memory=False)
X_val = val_data["sequence"].tolist()
MGW_val = val_data["MGW"].tolist()
EP_val = val_data["EP"].tolist()
ProT_val = val_data["ProT"].tolist()
Roll_val = val_data["Roll"].tolist()
HelT_val = val_data["HelT"].tolist()
y_val = val_data["label"].tolist()

test_data = pd.read_csv("/mnt/data/lly/LW/multi_se/data_with_shape/human/human_test_with_shape.bed", sep="\t", low_memory=False)
X_test = test_data["sequence"].tolist()
MGW_test = test_data["MGW"].tolist()
EP_test = test_data["EP"].tolist()
ProT_test = test_data["ProT"].tolist()
Roll_test = test_data["Roll"].tolist()
HelT_test = test_data["HelT"].tolist()
y_test = test_data["label"].tolist()

train_dataset = MyDataSet(X_train, MGW_train, EP_train, ProT_train, Roll_train, HelT_train, y_train)
val_dataset = MyDataSet(X_val, MGW_val, EP_val, ProT_val, Roll_val, HelT_val, y_val)
test_dataset = MyDataSet(X_test, MGW_test, EP_test, ProT_test, Roll_test, HelT_test, y_test)

train_dataloader = Data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = Data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = Data.DataLoader(test_dataset, batch_size=16, shuffle=False)

model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.GCN.parameters(), 'lr': 5e-5},
    {'params': model.Shape.parameters(), 'lr': 5e-5},
    {'params': [
        *model.BERT.parameters(),
        *model.AttPool.parameters(),
        *model.align_gcn.parameters(),
        *model.align_bert.parameters(),
        *model.align_shape.parameters(),
        *model.crossatt1.parameters(),
        *model.crossatt2.parameters(),
        *model.gate1.parameters(),
        *model.gate2.parameters(),
        *model.gate3.parameters(),
        *model.gate4.parameters(),
        *model.classifier.parameters(),
        model.temperature1,
        model.temperature2,
    ], 'lr': 1e-5}
])

epochs = 20
train_batch_size = 16
gradient_accumulation_steps = 1

total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
warmup_steps = 50 

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

best_val_auroc = 0  
best_model_path = args.best_model_path

# 用于存储每个 epoch 的训练和验证结果
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_mccs = []  # 存储训练集上的MCC
val_mccs = []  # 存储验证集上的MCC

# 记录训练开始时间
start_time = time.time()

patience = 3  # 如果验证集 AUROC 3 个 epochs 都没有提升，则停止训练
counter = 0   # 记录连续多少次没有提升

# 初始化混合精度工具
if args.use_amp:
    scaler = GradScaler()

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))

    # 训练过程
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    all_train_preds = []
    all_train_probs = []
    all_train_targets = []

    # 使用 tqdm 显示训练进度条
    for x1, a, b, x2, c, d, x3, y in tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}', leave=False):
        x1, a, b, x2, c, d, x3, y = x1.to(device), a.to(device), b.to(device), x2.to(device), c.to(device), d.to(device), x3.to(device), y.to(device)
        b = b[0]
        optimizer.zero_grad()  # 清除梯度

        # 根据use_amp控制混合精度开关
        if args.use_amp:
            with autocast():
                pred, g_proj, b_proj, s_proj = model(x1, a, b, x2, c, d, x3)
                loss = loss_fn(pred, y)

                sim_matrix_gb = torch.matmul(g_proj, b_proj.T) / model.temperature1
                labels_sim_gb = torch.arange(sim_matrix_gb.size(0)).to(sim_matrix_gb.device)
                loss_contrast_gb_1 = F.cross_entropy(sim_matrix_gb, labels_sim_gb)
                loss_contrast_gb_2 = F.cross_entropy(sim_matrix_gb.T, labels_sim_gb)
                loss_contrast_gb = (loss_contrast_gb_1 + loss_contrast_gb_2) / 2

                sim_matrix_bs = torch.matmul(b_proj, s_proj.T) / model.temperature2
                labels_sim_bs = torch.arange(sim_matrix_bs.size(0)).to(sim_matrix_bs.device)
                loss_contrast_bs_1 = F.cross_entropy(sim_matrix_bs, labels_sim_bs)
                loss_contrast_bs_2 = F.cross_entropy(sim_matrix_bs.T, labels_sim_bs)
                loss_contrast_bs = (loss_contrast_bs_1 + loss_contrast_bs_2) / 2

                lambda_contrast = 0.1
                loss = loss + lambda_contrast * loss_contrast_gb + lambda_contrast * loss_contrast_bs

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, g_proj, b_proj, s_proj = model(x1, a, b, x2, c, d, x3)
            loss = loss_fn(pred, y)

            sim_matrix_gb = torch.matmul(g_proj, b_proj.T) / model.temperature1
            labels_sim_gb = torch.arange(sim_matrix_gb.size(0)).to(sim_matrix_gb.device)
            loss_contrast_gb_1 = F.cross_entropy(sim_matrix_gb, labels_sim_gb)
            loss_contrast_gb_2 = F.cross_entropy(sim_matrix_gb.T, labels_sim_gb)
            loss_contrast_gb = (loss_contrast_gb_1 + loss_contrast_gb_2) / 2

            sim_matrix_bs = torch.matmul(b_proj, s_proj.T) / model.temperature2
            labels_sim_bs = torch.arange(sim_matrix_bs.size(0)).to(sim_matrix_bs.device)
            loss_contrast_bs_1 = F.cross_entropy(sim_matrix_bs, labels_sim_bs)
            loss_contrast_bs_2 = F.cross_entropy(sim_matrix_bs.T, labels_sim_bs)
            loss_contrast_bs = (loss_contrast_bs_1 + loss_contrast_bs_2) / 2

            lambda_contrast = 0.1
            loss = loss + lambda_contrast * loss_contrast_gb + lambda_contrast * loss_contrast_bs

            loss.backward()
            optimizer.step()

        scheduler.step()        

        train_loss += loss.item()
        _, train_predicted = torch.max(pred, 1)
        train_total += y.size(0)
        train_correct += (train_predicted == y).sum().item()

        all_train_preds.extend(train_predicted.cpu().numpy())
        all_train_targets.extend(y.cpu().numpy())

        train_probs = torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()  # 获取正类的概率
        all_train_probs.extend(train_probs)

    avg_train_loss = train_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total
    train_mcc = matthews_corrcoef(all_train_targets, all_train_preds)
    train_auroc = roc_auc_score(all_train_targets, all_train_probs)
    train_auprc = average_precision_score(all_train_targets, all_train_probs)
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(all_train_targets, all_train_preds).ravel()
    train_sensitivity = train_tp / (train_tp + train_fn) if (train_tp + train_fn) != 0 else 0
    train_specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) != 0 else 0

    print(f'Train Loss: {avg_train_loss: .4f}')
    print(f'Train Accuracy: {train_accuracy: .4f}%, Train MCC: {train_mcc: .4f}')
    print(f'Train SN: {train_sensitivity: .4f}, Train SP: {train_specificity: .4f}')
    print(f'Train AUROC: {train_auroc: .4f}, Train AUPRC: {train_auprc: .4f}')
    print('--------------------------------------------------------')
    # 保存训练集上的损失和准确率
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    train_mccs.append(train_mcc)  # 保存训练集上的MCC

    # 验证过程
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_probs = []
    all_val_targets = []

    # 使用 tqdm 显示验证进度条
    with torch.no_grad():
        for inputs1, a, b, inputs2, c, d, inputs3, target in tqdm(val_dataloader, desc=f'Validation Epoch {epoch + 1}', leave=False):
            inputs1, a, b, inputs2, c, d, inputs3, target = inputs1.to(device), a.to(device), b.to(device), inputs2.to(device), c.to(device), d.to(device), inputs3.to(device), target.to(device)
            b = b[0]
            output, g_proj, b_proj, s_proj = model(inputs1, a, b, inputs2, c, d, inputs3)
            loss = loss_fn(output, target)

            val_loss += loss.item()
            _, val_predicted = torch.max(output, 1)
            val_total += target.size(0)
            val_correct += (val_predicted == target).sum().item()

            all_val_preds.extend(val_predicted.cpu().numpy())
            all_val_targets.extend(target.cpu().numpy())

            val_probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # 获取正类的概率
            all_val_probs.extend(val_probs)

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = 100 * val_correct / val_total
    val_mcc = matthews_corrcoef(all_val_targets, all_val_preds)
    val_auroc = roc_auc_score(all_val_targets, all_val_probs)
    val_auprc = average_precision_score(all_val_targets, all_val_probs)
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(all_val_targets, all_val_preds).ravel()
    val_sensitivity = val_tp / (val_tp + val_fn) if (val_tp + val_fn) != 0 else 0
    val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) != 0 else 0

    print(f'Validation Loss: {avg_val_loss: .4f}')
    print(f'Validation Accuracy: {val_accuracy: .4f}%, Validation MCC: {val_mcc: .4f}')
    print(f'Validation SN: {val_sensitivity: .4f}, Validation SP: {val_specificity: .4f}')
    print(f'Validation AUROC: {val_auroc: .4f}, Validation AUPRC: {val_auprc: .4f}')
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    print(f'Current Learning Rate: {current_lr}')

    # 保存验证集上的损失和准确率
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    val_mccs.append(val_mcc)  # 保存验证集上的MCC

    # 验证过程结束后，检查早停
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save(model.state_dict(), best_model_path)  # 保存最佳模型
        print(f'Best model saved at epoch {epoch + 1} to {best_model_path}')
        counter = 0  # 有提升，重置计数器
    else:
        counter += 1
        print(f'No improvement in val_auroc for {counter} epoch(s).')
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}. Proceeding to test best model...')
            break

# 训练结束后，计算并打印总耗时
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time / 60:.2f} minutes")

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
