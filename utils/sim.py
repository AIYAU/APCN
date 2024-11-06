import numpy as np
import torch
import torch.nn.functional as F

# 获取设备，优先使用 CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从npy文件加载特征向量，并转移到GPU
tensor_list = torch.from_numpy(np.load("image_embeds.npy")).to(DEVICE)

def cosine_similarity_batch(tensor_a, tensor_b):
    # 确保 tensor_a 和 tensor_b 是2D张量：形状为 [batch_size, embedding_dim]
    if tensor_a.dim() == 1:
        tensor_a = tensor_a.unsqueeze(0)  # 如果是1D张量，增加一个批次维度
    if tensor_b.dim() == 1:
        tensor_b = tensor_b.unsqueeze(0)

    # 计算余弦相似度的点积部分
    dot_product = torch.matmul(tensor_a, tensor_b.T)  # 矩阵乘法，支持批量操作

    # 计算每个向量的 L2 范数
    norm_a = torch.norm(tensor_a, dim=1, keepdim=True)  # 形状为 [batch_size, 1]
    norm_b = torch.norm(tensor_b, dim=1, keepdim=True)  # 形状为 [batch_size, 1]

    # 使用广播机制计算归一化余弦相似度
    similarity = dot_product / (norm_a * norm_b.T)
    return similarity

def top_n_similarities_batch(target_tensors, n):
    similarities = []
    # 批量计算每个目标张量与所有支持张量的相似度
    for target_tensor in target_tensors:
        similarity_matrix = cosine_similarity_batch(target_tensor.unsqueeze(0), tensor_list)  # 形状为 [1, num_tensors]
        similarities.append(similarity_matrix.squeeze(0))  # 去掉第一个维度，形状为 [num_tensors]

    # 将相似度转换为张量
    similarities_tensor = torch.stack(similarities)

    # 获取相似度排名前 n 的索引
    top_n_indices = torch.argsort(similarities_tensor, dim=1, descending=True)[:, :n]

    # 提取前 n 的向量和对应的相似度
    top_n_tensors = torch.stack([tensor_list[indices] for indices in top_n_indices])
    top_n_similarities = torch.stack([similarities_tensor[i, indices] for i, indices in enumerate(top_n_indices)])

    # 对相似度进行 softmax 归一化
    softmax_similarities = F.softmax(top_n_similarities, dim=1)

    # 创建新向量，将前 n 个张量乘以它们的归一化相似度后相加
    new_vectors = torch.sum(top_n_tensors * top_n_similarities.unsqueeze(-1), dim=1)

    return new_vectors, top_n_tensors, softmax_similarities

if __name__ == '__main__':
    # 示例批量目标向量，转移到GPU
    target_tensors = torch.randn((5, 768)).to(DEVICE)  # 5个目标向量，每个向量有768维度

    # 设置需要返回的相似度排名前 n 的向量数量
    n = 3

    # 批量获取新向量以及前 n 个相似度最高的张量和它们的归一化相似度
    new_vectors, top_tensors, softmax_similarities = top_n_similarities_batch(target_tensors, n)

    # 输出结果
    print(f"新向量形状: {new_vectors.shape}")  # [batch_size, embedding_dim]
    print(f"相似度排名前 {n} 的向量及其归一化相似度:")
    print(f"Top Tensors Shape: {top_tensors.shape}")  # [batch_size, n, embedding_dim]
    print(f"Softmax Similarities Shape: {softmax_similarities.shape}")  # [batch_size, n]

    print(torch.cat([target_tensors, new_vectors], dim=1).shape)  # [batch_size, 2 * embedding_dim]
