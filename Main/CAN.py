import torch
import numpy as np
def CAN(y_pred,y_true,label_prior,top_k=2,threshold=0.991,alpha=1,iters=3):
    """
    在mydata好用的参数：k=6,threshold = 0.45
    先验概率修正输出概率
    :param y_pred: [batch_size,num_labels]  torch.tensor
    :param y_true: [batch_size]
    :param num_labels: [batch_size]
    :return:
    """
    # 定义这两个列表，因为经过熵筛选后标签顺序发生了变化，需要重新获取预测标签和原来的标签
    y_pred_label_list = []
    y_true_label_list = []
    y_pred = torch.softmax(torch.tensor(y_pred),dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    # acc_original = np.mean([y_pred.argmax(1) == y_true])
    # print('acc_original: %s' % acc_original)
    # prior = get_cate_prior()
    prior = label_prior
    k = top_k  # 6
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)  # 归一化
    y_pred_entropy = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)  # top-k熵
    # print(y_pred_entropy)
    # 选择阈值，划分高、低置信度两部分
    threshold = threshold  # 0.45
    y_pred_confident = y_pred[y_pred_entropy < threshold]  # top-k熵低于阈值的是高置信度样本
    y_pred_unconfident = y_pred[y_pred_entropy >= threshold]  # top-k熵高于阈值的是低置信度样本
    y_true_confident = y_true[y_pred_entropy < threshold]
    y_true_unconfident = y_true[y_pred_entropy >= threshold]
    # 显示两部分各自的准确率
    # 一般而言，高置信度集准确率会远高于低置信度的
    y_pred_label_list.extend(y_pred_confident.argmax(1).tolist())  # 存储预测的高置信度的标签顺序
    y_true_label_list.extend(y_true_confident) # 存储打乱顺序后的实际标签
    y_true_label_list.extend(y_true_unconfident)
    # acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
    # acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).float().mean()
    # print('confident acc: %s' % acc_confident)
    # print('unconfident acc: %s' % acc_unconfident)
    # 逐个修改低置信度样本，并重新评价准确率
    right, alpha, iters = 0, alpha, iters  # 正确的个数，alpha次方，iters迭代次数 alpha =1 iters = 5
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)  # Y is L_0
        for _ in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        index = int(y.argmax())
        y_pred_label_list.append(index)
        # if y.argmax() == y_true_unconfident[i]:
        #     right += 1
    # 输出修正后的准确率
    # acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
    # print('new unconfident acc: %s' % (right / (i + 1.)))
    # print('final acc: %s' % acc_final)
    return y_true_label_list,y_pred_label_list