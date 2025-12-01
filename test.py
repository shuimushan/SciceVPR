
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import os
import shutil


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))#database直接改成16，queries随infer_batch_size变####改回来了
        
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))#待用region descriptors
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    # 在计算recalls的循环之前初始化变量
    mrr = 0.0
    f1_scores = {n: 0.0 for n in args.recall_values}  # 为每个recall值存储F1分数
    for query_index, pred in enumerate(predictions):
        positives = positives_per_query[query_index]
        total_positives = len(positives)
    
        # 计算MRR
        first_correct_rank = None
        for rank, idx in enumerate(pred, 1):
            if idx in positives:
                first_correct_rank = rank
                break
        if first_correct_rank is not None:
            mrr += 1.0 / first_correct_rank
    
        # 计算每个recall值的F1-score
        for n in args.recall_values:
            tp = np.sum(np.in1d(pred[:n], positives))  # 前N个中的真正例
            precision = tp / n
            recall = tp / total_positives if total_positives > 0 else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores[n] += f1
            
        ##原本的recall@N
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    # 计算平均值
    mrr /= eval_ds.queries_num
    for n in args.recall_values:
        f1_scores[n] /= eval_ds.queries_num


    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    f1_str = ", ".join([f"F1@{val}: {f1:.3f}" for val, f1 in f1_scores.items()])
    mrr_str = f"MRR: {mrr:.3f}"

    full_stats_str = recalls_str + ", " + f1_str + ", " + mrr_str
    return recalls, full_stats_str#, mrr , f1_scores


