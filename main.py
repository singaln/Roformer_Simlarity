# coding=utf-8
# @Time:2021/6/215:14
# @author: SinGaln

import argparse
from trainer import Trainer
from data_loader import load_and_cache_examples
from utils import init_logger, load_tokenizer, set_seed

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_data = load_and_cache_examples(args, tokenizer, mode="train")
    test_data = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_data, test_data)

    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The path of data.")
    parser.add_argument("--task", default="similar", type=str, help="The name of task.")
    parser.add_argument("--label_file", default="label.txt",type=str, help="label file path.")
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model name.")
    parser.add_argument("--model_dir", default="./similar_bert", type=str, help="Save path of new model.")
    parser.add_argument("--pretrained_model_path", default="./chinese_bert_wwm", type=str, help="Pretrained model path.")
    parser.add_argument("--seed", default=1234, type=int, help="The seed of random.")
    parser.add_argument("--max_seq_len", default=100, type=int, help="The max sequence length of data.")
    parser.add_argument("--ignore_index", default=0, type=int, help="Specifies a target value that is ignored and does not distribute to the input gradient.")
    parser.add_argument("--embedding_size", default=768, type=int, help="Embedding size of input data.")
    parser.add_argument("--num_attention_heads", default=12, type=int, help="The number of attention heads.")
    parser.add_argument("--attention_dropout_prob", default=0.1, type=float, help="The dropout rate of multi attention model.")
    parser.add_argument("--hidden_size", default=1024, type=int, help="The hidden size of model middle layer.")
    parser.add_argument("--feed_dropout_rate", default=0.1, type=int, help="The dropout rate of feed forward layer.")
    parser.add_argument("--max_steps", default=-1, type=int, help="Set total number of training steps to perform.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backwrd pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="The epsilon value of Adam.")
    parser.add_argument("--warm_up", default=0, type=int, help="Linear warmup over warmup steps.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--logging_steps", default=1000, type=int, help="Log every x updates steps.")
    parser.add_argument("--save_steps", default=999, type=int, help="Save checkpoint every x updates steps.")
    parser.add_argument("--vocab_size", default=8007, type=int, help="The size for vocab.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluate.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluate.")

    args = parser.parse_args()
    main(args)
