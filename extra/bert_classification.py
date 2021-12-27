# %%
import jieba
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tqdm
import random
import csv
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AdamW, AutoTokenizer, get_linear_schedule_with_warmup

# %%


class BertClassificationModel:
    """创建分类模型类，使用transforms库中的BertForSequenceClassification类，并且该类自带loss计算"""

    def __init__(self, train, validation, vocab_path, config_path, pretrain_model_path, save_model_path,
                 learning_rate, n_class, epochs, batch_size, val_batch_size, max_len, gpu=True):
        super(BertClassificationModel, self).__init__()
        # 类别数
        self.n_class = n_class
        # 句子最大长度
        self.max_len = max_len
        # 学习率
        self.lr = learning_rate
        # 加载训练数据集
        self.train = self.load_data(train)
        # 加载测试数据集
        self.validation = self.load_data(validation)
        # 训练轮数
        self.epochs = epochs
        # 训练集的batch_size
        self.batch_size = batch_size
        # 验证集的batch_size
        self.val_batch_size = val_batch_size
        # 模型存储位置
        self.save_model_path = save_model_path
        # 是否使用gpu
        self.gpu = gpu

        # 加载bert分词模型词典
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        # 加载bert模型配置信息
        config = BertConfig.from_json_file(config_path)
        # 设置分类模型的输出个数
        config.num_labels = n_class
        # 加载bert分类模型
        self.model = BertForSequenceClassification.from_pretrained(
            pretrain_model_path, config=config)
        # 设置GPU
        if self.gpu:
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

    def encode_fn(self, text_lists):
        """
        训练：将text_list embedding成bert模型可用的输入形式
        :param text_lists:['我爱你','猫不是狗']
        :return:
        """
        # 返回的类型为pytorch tensor
        tokenizer = self.tokenizer(
            text_lists,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def load_data(self, path):
        """
        训练：处理训练的csv文件
        :param path:
        :return:
        """
        text_lists = []
        labels = []
        for line in csv.reader(open(path, encoding='utf-8')):
            # 这里可以改，label在什么位置就改成对应的index
            label = int(line[0])
            text = line[1]
            text_lists.append(text)
            labels.append(label)
        input_ids, token_type_ids, attention_mask = self.encode_fn(text_lists)
        labels = torch.tensor(labels)
        data = Dataset(input_ids, token_type_ids, attention_mask, labels)
        return data

    def train_model(self):
        """
        训练：训练模型
        :return:
        """
        if self.gpu:
            self.model.cuda()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # 处理成多个batch的形式
        train_data = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True)
        val_data = DataLoader(
            self.validation,
            batch_size=self.val_batch_size,
            shuffle=True)

        total_steps = len(train_data) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss, total_val_loss = 0, 0
            total_eval_accuracy = 0
            print('epoch:', epoch, ', step_number:', len(train_data))
            # 训练，其中step是迭代次数
            # 每一次迭代都是一次权重更新，每一次权重更新需要batch_size个数据进行Forward运算得到损失函数，再BP算法更新参数。
            # 1个iteration等于使用batch_size个样本训练一次
            for step, batch in enumerate(train_data):
                self.model.zero_grad()
                # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     token_type_ids=batch[1].to(self.device),
                                     attention_mask=batch[2].to(self.device),
                                     labels=batch[3].to(self.device))
                loss, logits = outputs[0], outputs[1]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                # 每100步输出一下训练的结果，flat_accuracy()会对logits进行softmax
                if step % 100 == 0 and step > 0:
                    self.model.eval()
                    logits = logits.detach().cpu().numpy()
                    label_ids = batch[3].cuda().data.cpu().numpy()
                    avg_val_accuracy = self.flat_accuracy(logits, label_ids)
                    print(f'step:{step}')
                    print(f'Accuracy: {avg_val_accuracy:.4f}')
                    print('*'*20)
            # 每个epoch结束，就使用validation数据集评估一次模型
            self.model.eval()
            print('testing ....')
            for i, batch in enumerate(val_data):
                with torch.no_grad():
                    outputs = self.model(input_ids=batch[0].to(self.device),
                                         token_type_ids=batch[1].to(
                                             self.device),
                                         attention_mask=batch[2].to(
                                             self.device),
                                         labels=batch[3].to(self.device))
                    loss, logits = outputs[0], outputs[1]
                    total_val_loss += loss.item()

                    logits = logits.detach().cpu().numpy()
                    label_ids = batch[3].cuda().data.cpu().numpy()
                    total_eval_accuracy += self.flat_accuracy(
                        logits, label_ids)

            avg_train_loss = total_loss / len(train_data)
            avg_val_loss = total_val_loss / len(val_data)
            avg_val_accuracy = total_eval_accuracy / len(val_data)

            print(f'Train loss     : {avg_train_loss}')
            print(f'Validation loss: {avg_val_loss}')
            print(f'Accuracy: {avg_val_accuracy:.4f}')
            print('*'*20)
            self.save_model(self.save_model_path + '-' + str(epoch))

    def save_model(self, path):
        """
        训练：保存分词模型和分类模型
        :param path:
        :return:
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def load_model(path):
        """
        预测：加载分词模型和分类模型
        :param path:
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path)
        return tokenizer, model

    @staticmethod
    def load_data_predict(path):
        """
        预测：加载测试数据
        :param path:
        :return:
        """
        text_lists = []
        labels = []
        for line in csv.reader(open(path, encoding='utf-8')):
            text = line[1]
            text_lists.append(text)
            label = int(line[0])
            labels.append(label)
        return text_lists, labels

    def eval_model(self, tokenizer, model, text_lists, y_real):
        """
        预测：输出模型的召回率、准确率、f1-score
        :param tokenizer:
        :param model:
        :param text_lists:
        :param y_real:
        :return:
        """
        preds = self.predict_batch(tokenizer, model, text_lists)
        print(classification_report(y_real, preds))

    def predict_batch(self, tokenizer, model, text_lists):
        """
        预测：预测
        :param tokenizer:
        :param model:
        :param text_lists:
        :return:
        """
        tokenizer = tokenizer(
            text_lists,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        pred_data = Dataset(input_ids, token_type_ids, attention_mask)
        pred_dataloader = DataLoader(
            pred_data, batch_size=self.batch_size, shuffle=False)
        model = model.to(self.device)
        model.eval()
        preds = []
        for i, batch in enumerate(pred_dataloader):
            with torch.no_grad():
                outputs = model(input_ids=batch[0].to(self.device),
                                token_type_ids=batch[1].to(self.device),
                                attention_mask=batch[2].to(self.device)
                                )
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                preds += list(np.argmax(logits, axis=1))
        return preds


if __name__ == '__main__':
    EPOCH = 3
    # 预训练模型的存储位置为 ../../pretrained_models/bert-base-chinese/
    # 分类模型和分词模型的存储位置是 ../trained_model/bert_model/
    bert_model = BertClassificationModel(
        train='./data/train.csv',
        validation='./data/validation.csv',
        vocab_path='./pretrained_models/bert-base-chinese/vocab.txt',
        config_path='./pretrained_models/bert-base-chinese/bert_config.json',
        pretrain_model_path='./pretrained_models/bert-base-chinese/pytorch_model.bin',
        save_model_path='./trained_model/bert_model',
        learning_rate=2e-5,
        n_class=723,
        epochs=EPOCH,
        batch_size=4,
        val_batch_size=4,
        max_len=50,
        gpu=True)
    # 模型训练
    bert_model.train_model()

    # 模型预测
    classification_tokenizer, classification_model = bert_model.load_model(
        bert_model.save_model_path + '-' + str(EPOCH - 1))
    text_list, y_true = bert_model.load_data_predict('./data/validation.csv')
    bert_model.eval_model(classification_tokenizer,
                          classification_model, text_list, y_true)
