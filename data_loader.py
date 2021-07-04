# coding=utf-8
# @Time:2021/6/210:11
# @author: SinGaln

"""处理加载数据"""
import os
import ast
import copy
import json
import torch
import logging
from utils import get_labels
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    输入的为两个句子，利用[SEP]进行分隔
    Args:
        guid: 实例的唯一id
        text_a: 句对中的第一个句子
        text_b: 句对中的第二个句子
        label: 实例对象的标签
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels, input_length):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_length = input_length
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """对输入的数据进行处理"""

    def __init__(self, args):
        self.args = args
        self.labels = get_labels(args)

        self.input_text_file = "data.txt"

    @classmethod
    def _read_file(cls, input_file):
        return_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            data_lst = f.readlines()
            for sentences in data_lst:
                sentences_dict = ast.literal_eval(sentences)
                query_text = sentences_dict.get("query", False)
                condition_lst = sentences_dict.get("candidate", False)
                for condition_dic in condition_lst:
                    condition_text = condition_dic.get("text", False)
                    condition_label = condition_dic.get("label", False)
                    return_list.append((query_text, condition_text, condition_label))
        return return_list

    def _create_examples(self, texts, set_type):
        """创建训练, 测试数据的实例"""
        examples = []
        for i, data in enumerate(texts):
            guid = "%s-%s" % (set_type, i)
            # query的文本内容
            words = [word for word in data[0]]
            # condition_text的文本内容
            condition_text = [cond for cond in data[1]]
            # 标签转为id
            labels = self.labels.index(data[2]) if data[2] in self.labels else self.labels.index("UNK")
            examples.append(InputExample(guid=guid, text_a=words, text_b=condition_text, label=labels))
        return examples

    def get_examples(self, mode):
        """
        更加mode返回相应的训练,测试数据
        Args:
            mode:train, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("数据加载 {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     set_type=mode)


processors = {
    "similar": DataProcessor
}

def concat_seq_pair(tokens_a, tokens_b, max_seq_len):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length < max_seq_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_segment_id=1):
    # 基本设置
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (example_index, example) in enumerate(examples):
        if example_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (example_index, len(examples)))
        text_a = example.text_a
        text_b = example.text_b
        labels = example.label

        tokens_a = []
        tokens_b = []
        for word in text_a:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens_a.extend(word_tokens)

        for word in text_b:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens_b.extend(word_tokens)

        concat_seq_pair(tokens_a, tokens_b, max_seq_len)
        tokens = []
        token_type_ids = []
        tokens.append(cls_token)
        token_type_ids.append(cls_token_segment_id)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(sequence_a_segment_id)
        tokens.append(sep_token)
        token_type_ids.append(cls_token_segment_id)

        for token in tokens_b:
            tokens.append(token)
            token_type_ids.append(sequence_b_segment_id)
        tokens.append(sep_token)
        token_type_ids.append(sep_token_segment_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_token_id)
            attention_mask.append(pad_token_id)
            token_type_ids.append(pad_token_id)
        # # 先拼接再padding
        # tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # if len(input_ids) > max_seq_len:
        #     input_ids = input_ids[:(max_seq_len - len(input_ids))]
        # else:
        #     input_ids = input_ids + (max_seq_len - len(input_ids)) * [pad_token_id]
        # # print(input_ids, len(input_ids))
        # token_type_ids = [cls_token_segment_id] + (len(tokens_a) * [sequence_a_segment_id]) + [cls_token_segment_id] + \
        #                  (max_seq_len - len([cls_token_segment_id] + (len(tokens_a) * [sequence_a_segment_id]) + [cls_token_segment_id])) * [sep_token_segment_id]
        # # print(token_type_ids, len(token_type_ids))
        # attention_mask = []
        # for i in input_ids:
        #     if i != 0:
        #         attention_mask.append(1)
        #     else:
        #         attention_mask.append(0)
        # print(attention_mask, len(attention_mask))
        # special_token_count = 2
        # if len(tokens_a) > max_seq_len - special_token_count:
        #     tokens_a = tokens_a[:(max_seq_len - special_token_count)]
        #
        # # 增加[SEP]和[CLS]
        # tokens_a += [sep_token]
        # token_a_type_ids = [sequence_a_segment_id] * len(tokens_a)
        # tokens_b += [sep_token]
        # token_b_type_ids = [sequence_b_segment_id] * len(tokens_b)
        # # 增加[CLS]
        # tokens_a = [cls_token] + tokens_a
        # token_a_type_ids = [cls_token_segment_id] + token_a_type_ids
        # # 拼接token_a和token_b
        # tokens = tokens_a + tokens_b
        # token_type_ids = token_a_type_ids + token_b_type_ids
        #
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # attention mask 用1表示真实token, 0表示padding token

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask {} vs {}".format(len(attention_mask),
                                                                                               max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type {} vs {}".format(len(token_type_ids),
                                                                                           max_seq_len)
        label_id = int(labels)

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: % s" % " ".join([str(x) for x in tokens]))
            logger.info("inputs_ids: % s" % " ".join([str(x) for x in input_ids]))
            logger.info("token_type_ids: % s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("attention_mask: % s" % " ".join([str(x) for x in attention_mask]))
            logger.info("labels: % s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      labels=label_id, input_length=len(input_ids)))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    cached_features_file = os.path.join(
        args.data_dir,
        "{}_{}_{}_{}".format(
            mode,
            args.task,
            list(filter(None, args.pretrained_model_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("The mode only include train, test!")

        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, pad_token_label_id)

        logger.info("Save features into cache file %s", cached_features_file)
        torch.save(features, cached_features_file)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)
    return dataset
