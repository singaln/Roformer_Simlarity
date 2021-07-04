# coding=utf-8
# @Time:2021/6/29:45
# @author: SinGaln

"""utils文件"""
import os
import torch
import random
import logging
import numpy as np
from model import BertModelOutputs
from transformers import BertConfig, BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_CLASSES = {
    "bert":(BertConfig, BertModelOutputs, BertTokenizer)
}

MODEL_PATH_MAP = {
    "bert":"./chinese_bert_wwm"
}

# 获取label(完全匹配, 部分匹配, 不匹配)
def get_labels(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.task, args.label_file), "r", encoding="utf-8")]

# 加载tokenizer
def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.pretrained_model_path)

# 设置logger
def init_logger():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO)

# 设置种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# 计算precision_score, recall_score, f1_score
def get_metrics(pred_label, true_label):
    assert len(pred_label) == len(true_label)
    return {
        "precision_score":precision_score(true_label, pred_label, average="macro"),
        "recall_score": recall_score(true_label, pred_label, average="macro"),
        "f1": f1_score(true_label, pred_label, average="macro")
    }
