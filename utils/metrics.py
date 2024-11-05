import cv2
import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def show_images(query_image_path, gallery_image_paths):
    print("===========================queryimage================================")
    print(query_image_path)
    # query_image = cv2.imread(query_image_path)
    # if query_image is None:
    #     print(f"Warning: could not read query image {query_image_path}")
    #     return
    #
    # cv2.imshow("Query Image", query_image)

    print("===========================gimage================================")
    for i, gallery_image_path in enumerate(gallery_image_paths):
        print(gallery_image_path)
    #     gallery_image = cv2.imread(gallery_image_path)
    #     if gallery_image is None:
    #         print(f"Warning: could not read gallery image {gallery_image_path}")
    #         continue
    #     cv2.imshow(f"Gallery Image {i+1}", gallery_image)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    # 对于每个query图像，将其相同摄像头视角下的gallery图像舍弃
    num_q, num_g = distmat.shape

    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # 使用numpy中的argsort函数对距离进行排序，按行进行排序
    # 即每行元素的排序结果会被返回，形成新的索引数组
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0

    # 创建匹配矩阵，用于标识图库图像是否与查询图像匹配
    # 利用索引数组indices，在图库图像标签数组g_pids上进行索引
    # q_pids[:, np.newaxis]将查询图像标签数组转换成列向量，方便与图库图像标签数组进行比较
    # 最终将比较结果转换成整数类型，表示匹配(1)或不匹配(0)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []

    num_valid_q = 0.  # 有效查询数量
    for q_idx in range(num_q):
        # 获取查询pid和camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # 删除与查询具有相同pid和camid的图库样本
        order = indices[q_idx]  # 选择一行
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # 计算cmc曲线
        # 二进制向量，值为1的位置是正确匹配项
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # 当查询身份不出现在图库中时，此条件为真
            continue

        # 计算累积匹配数
        cmc = orig_cmc.cumsum()
        # 将累积匹配数中大于1的部分截断为1，确保cmc只包含0和1，表示是否匹配
        cmc[cmc > 1] = 1

        # 将cmc添加到所有查询的cmc列表中，限制最大长度为max_rank
        all_cmc.append(cmc[:max_rank])
        # 增加有效查询数量
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        # 计算相关样本数量（即匹配项的总数）
        num_rel = orig_cmc.sum()
        # 计算累积匹配数
        tmp_cmc = orig_cmc.cumsum()

        # 计算平均准确率（AP）
        # 根据公式AP=sum(Precision * ΔRecall)，其中ΔRecall为Recall的增量
        # 通过tmp_cmc计算每个阈值下的Precision值，再乘以相应的Recall增量，得到AP
        # 具体而言，将每个阈值下的Precision除以相应的Recall增量，得到的结果为Precision-Recall曲线上的点，然后将这些点的值乘以匹配项的二进制值，再求和即可得到AP
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        # # 获取rank-10的gallery图像路径
        # top10_indices = order[:15]
        # top10_gallery_image_paths = [g_image_paths[i] for i in top10_indices]
        # show_images(q_image_paths[q_idx], top10_gallery_image_paths)

    assert num_valid_q > 0, "错误：所有查询身份在图库中都不出现"

    # 将所有查询的cmc列表转换为numpy数组，并转换数据类型为np.float32
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    # 对所有查询的cmc曲线进行求和，然后除以有效查询数量，得到平均cmc曲线
    all_cmc = all_cmc.sum(0) / num_valid_q
    # 计算所有查询的平均准确率（mAP）
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # self.image_paths = []  # 初始化image_paths

    # 在每个epoch结束后调用
    def update(self, output):  # called once for each batch
        # feat, pid, camid, image_paths = output
        feat, pid, camid  = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        # 将所有特征拼接在一起
        feats = torch.cat(self.feats, dim=0)

        # 如果需要对特征进行归一化
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # 查询特征和对应的pid、相机id
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # q_image_paths = self.image_paths[:self.num_query]

        # 库特征和对应的pid、相机id
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        # g_image_paths = self.image_paths[self.num_query:]

        # 如果需要重新排序
        if self.reranking:
            print('=> Enter reranking') # 输出提示信息
            # 使用重新排序的方法计算距离矩阵
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance') # 输出提示信息
            # 使用欧氏距离计算距离矩阵
            distmat = euclidean_distance(qf, gf)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_image_paths, g_image_paths)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)


        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



