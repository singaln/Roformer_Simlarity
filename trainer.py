# coding=utf-8
# @Time:2021/6/213:42
# @author: SinGaln

"""训练文件"""
import os
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
from utils import MODEL_CLASSES, get_metrics, get_labels
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_data=None, test_data=None):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.label_lst = get_labels(args)

        self.pad_token_label_id = args.ignore_index
        # 模型初始化
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.configs = self.config_class.from_pretrained(args.pretrained_model_path,
                                                         finetuning_task=self.args.task)
        self.model = self.model_class.from_pretrained(args.pretrained_model_path,
                                                      args=args,
                                                      config=self.configs)

        # 设别选择(GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # 多GPU
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train(self):
        train_sampler = RandomSampler(self.train_data)
        train_loader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_loader) // self.args.gradient_accumulation_steps) + 1
        else:
            total_steps = len(train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # optimizer and schedule
        no_decay = ["bias", "LayerNorm.weight"]
        # optimizer parameters setting
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warm_up,
                                                   num_training_steps=total_steps)

        # train information
        logger.info("********** Running Training **********")
        logger.info("num example = %d", len(self.train_data))
        logger.info("num epochs = %d", self.args.num_train_epochs)
        logger.info("train batch size = %d", self.args.train_batch_size)
        logger.info("gradient accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("total steps = %d", total_steps)
        logger.info("logger steps = %d", self.args.logging_steps)
        logger.info("save steps = %d", self.args.save_steps)

        global_steps = 0
        tr_loss = 0.0
        loss_fun = torch.nn.CrossEntropyLoss()
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_loader)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_id = batch
                # print(input_ids)
                # print(attention_mask)
                # print(token_type_ids)
                # print(label_id)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # print(output[-1])
                # print(label_id.view(-1))
                loss = loss_fun(output.view(-1, len(self.label_lst)), label_id)
                print()
                print("loss",loss)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                else:
                    loss.backward()
                tr_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()
                    global_steps += 1

                    if self.args.logging_steps > 0 and global_steps % self.args.logging_steps == 0:
                        self.evaluate("test")
                    if self.args.save_steps > 0 and global_steps % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_steps:
                    epoch_iterator.close()
                    break
        return global_steps, tr_loss / global_steps

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_data
        else:
            raise Exception("The dataset is not existing!")

        eval_sampler = SequentialSampler(dataset)
        eval_loader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # eval logging
        logger.info("********** logger information **********")
        logger.info("num example = %d", len(dataset))
        logger.info("batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        eval_steps = 0
        loss_fun = torch.nn.CrossEntropyLoss()
        label_preds = None
        label_ids = None

        self.model.eval()
        for batch in tqdm(eval_loader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, token_type_ids, label_id = batch
                outputs = self.model(input_ids, attention_mask, token_type_ids)

                eval_loss = loss_fun(outputs.view(-1, len(self.label_lst)), label_id)
                eval_loss + eval_loss.mean().item()
            eval_steps += 1

            # predict
            if label_preds is None:
                label_preds = outputs.detach().cpu().numpy()
                label_ids = label_id.detach().cpu().numpy()
            # else:
            #     label_preds = np.append(label_preds, outputs.detach().cpu().numpy(), axis=0)
            #     label_ids = np.append(label_ids, label_id.detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / eval_steps
        results = {
            "loss": eval_loss
        }

        label_preds = np.argmax(label_preds, axis=1)
        print("label_preds", label_preds, len(label_preds))
        print("label_ids", label_ids, len(label_ids))
        total_result = get_metrics(label_preds, label_ids)
        results.update(total_result)

        logger.info("********** Evaluate Results **********")
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))
        return results

    def save_model(self):
        # 模型保存
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # 保存模型训练的超参
        torch.save(self.args, os.path.join(self.args.model_dir, "train_args.bin"))
        logger.info("model parameters save %s", self.args.model_dir)

    def load_model(self):
        # 加载模型
        if os.path.exists(self.args.model_dir):
            raise Exception("The model is not existing!")
        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args)
            self.model.to(self.device)
            logger.info("********** model load success **********")
        except:
            raise Exception("The model lost or damage!")
