import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置随机种子以确保实验的可重复性
random_seed = [42,96,128,45232]
all_result = []

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_way = 3
n_shot = 10
n_query = 20

DEVICE = "cuda"
n_workers = 12
data_name = 'K8000'
from easyfsl.datasets import CUB, EYE, OCT,K500,K7000,K8000,K700
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader

n_tasks_per_epoch = 500
n_validation_tasks = 40

train_set = K7000(split="train", training=True, image_size=224)
val_set = K7000(split="val", training=False, image_size=224)

train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch)
val_sampler = TaskSampler(val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks)

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

from easyfsl.methods import ClipPrototypicalNetworks, ClipGPT2PrototypicalNetworks
from easyfsl.modules import resnet12, VSSM

# backbone_name = 'resnet10t'swin_s3_small_224
backbone_name = 'mamba'
# backbone_name = 'swin224'
# backbone_name = 'resnet12'
# model_name = 'ClipGPT2PrototypicalNetworks'swin_s3_small_224
model_name = 'ClipPrototypicalNetworks'
# model_name = 'ClipBERTT2PrototypicalNetworks'swin_small_patch4_window7_224
import timm


# convolutional_network = timm.create_model('resnet10t', pretrained=False,num_classes = 768)
convolutional_network = timm.create_model('swin_s3_small_224', pretrained=True,num_classes = 768)
#convolutional_network = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=1000).to(DEVICE)  # 主干Mamba
# convolutional_network = resnet12()
convolutional_network = nn.DataParallel(convolutional_network)  # mamba不能开平行
few_shot_classifier = ClipPrototypicalNetworks(backbone = convolutional_network).to(DEVICE)  # 使用对比原型,加原型修正
# few_shot_classifier = ClipGPT2PrototypicalNetworks(convolutional_network).to(DEVICE)  # 使用对比原型+文本解码器
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
LOSS_FUNCTION = nn.CrossEntropyLoss()
n_epochs = 2
scheduler_milestones = [1,4,6,8,10]
scheduler_gamma = 0.1
learning_rate = 1e-4
tb_logs_dir = Path("./logs")

train_optimizer = torch.optim.Adam(few_shot_classifier.parameters(), lr=learning_rate, weight_decay=5e-4)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer, milestones=scheduler_milestones,gamma=scheduler_gamma)
tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

def training_epoch(model, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _,) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(support_images.to(DEVICE), support_labels.to(DEVICE))
            classification_scores = model(query_images.to(DEVICE))
            loss_l = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss = loss_l + model.contrast_loss  # 交叉损失加对比损失
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)


from easyfsl.utils import evaluate

for seed in random_seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0

    print(f'{model_name}-{data_name}-{backbone_name}-{n_way}way-{n_shot}shot')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation")

        if validation_accuracy["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy["accuracy"]
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy["accuracy"], epoch)

        train_scheduler.step()

    few_shot_classifier.load_state_dict(best_state)

    n_test_tasks = 40

    test_set = K7000(split="val", training=False, image_size=224)
    test_sampler = TaskSampler(test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks)
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
