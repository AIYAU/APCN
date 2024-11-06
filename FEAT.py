import copy
from pathlib import Path
import random
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm

# 设置随机种子以确保实验的可重复性

# 设置随机种子以确保实验的可重复性
random_seed = [42,96,128,45232]
all_result = []

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 定义few-shot-learning的参数
n_way = 3
n_shot = 10
n_query = 5
DEVICE = "cuda"
n_workers = 8

from easyfsl.datasets import CUB, EYE, OCT,K500,K7000,K8000,K700
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader

# 定义每轮训练和验证的任务数量
n_tasks_per_epoch = 100
n_validation_tasks = 40

# 实例化数据集
train_set = K8000(split="train", training=True, image_size=224)
val_set = K8000(split="val", training=False, image_size=224)
print(f"Number of unique labels in training set: {len(set(train_set.labels))}")
print(f"Number of unique labels in validation set: {len(set(val_set.labels))}")

# 实例化采样器，用于生成few-shot任务
train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch)
val_sampler = TaskSampler(val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks)
# 实例化DataLoader，用于加载数据
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

# 导入few-shot学习方法和模型
from easyfsl.methods import FEAT, FewShotClassifier
from easyfsl.modules import MultiHeadAttention, resnet12

from torchvision.models import resnet18

# convolutional_network = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
convolutional_network = resnet18()
# convolutional_network = resnet12()
convolutional_network.fc = nn.Identity()
# 编码维度和关系提取模块维度一致
# convolutional_network.fc = nn.Identity()
# print(convolutional_network)


convolutional_network = nn.DataParallel(convolutional_network)  # mamba不能开平行

few_shot_classifier = FEAT(backbone=convolutional_network, attention_module=MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1)).to(DEVICE)

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

LOSS_FUNCTION = nn.CrossEntropyLoss()
# 定义训练参数
n_epochs = 1
scheduler_milestones = [1,4,6,8,10]
scheduler_gamma = 0.1
learning_rate = 1e-5
tb_logs_dir = Path("./logs")
train_optimizer = Adam(few_shot_classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
# 实例化学习率调度器
train_scheduler = MultiStepLR(
    train_optimizer,
    milestones=scheduler_milestones,
    gamma=scheduler_gamma,
)
# 实例化TensorBoard写入
tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
def training_epoch(model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model.train()
    # print(next(iter(data_loader)), type(next(iter(data_loader))))
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set( # 先处理支持集，得到支持集的原型表示
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            classification_scores = model(query_images.to(DEVICE)) # 将查询集喂入模型

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

# 导入评估工具
from easyfsl.utils import evaluate
# 在测试集上评估模型性能
for seed in random_seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # 初始化最佳模型状态和验证准确率
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )

        if validation_accuracy["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy["accuracy"]
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models
            print("Ding ding ding! We found a new best model!")
            print(validation_accuracy)

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy["accuracy"], epoch)

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    # 加载最佳模型权重
    few_shot_classifier.load_state_dict(best_state)

    # 定义测试任务数量
    n_test_tasks = 40

    # 实例化测试数据集和采样器
    test_set = K8000(split="val", training=False, image_size=224)
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    # 在测试集上评估模型性能
    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
    all_result.append(accuracy)

# 将字典列表转换为DataFrame
result_df = pd.DataFrame(all_result)
# 计算每个指标的均值和标准差
means = result_df.mean()
stds = result_df.std()
# 将均值和标准差组合成 "mean ± std" 的格式
result = pd.DataFrame({
    'mean': means,
    'std': stds,
    'mean ± std': means.round(6).astype(str) + ' ± ' + stds.round(6).astype(str)
})
print(result)