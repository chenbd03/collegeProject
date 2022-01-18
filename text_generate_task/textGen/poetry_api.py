#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import os
from keras.layers import LSTM, Dropout, Dense,Embedding,GRU,Activation
from keras.models import Input, Model, load_model,Sequential
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback,ModelCheckpoint
# from keras.layers.recurrent import LSTM, GRU



# 定义配置类
class ModelConfig(object):
    poetry_file = os.getcwd() + os.sep + "textGen" + os.sep + 'poetry.txt'  # 数据集
    weight_file = os.getcwd() + os.sep + "textGen" + os.sep +'poetry_model.h5'  # 模型保持路径
    max_len = 6   # 五言绝句
    batch_size = 64  # pi
    learning_rate = 0.0003

# 定义文件读取函数
def preprocess_data(ModelConfig):
    # print("读取五言律诗...")
    # 语料文本内容
    files_content = ''
    count = 0
    with open(ModelConfig.poetry_file, 'r',encoding='UTF-8') as f:
        for line in f:
            x = line.strip() + "]"    # 因为每首诗的长度不一样，所以在这里加一个分隔
            # 取出具体诗的内容
            x = x.split(":")[1]
            # 根据长度过滤脏数据
            if len(x) <= 5 :
                continue
            # 过滤出五言绝句
            if x[5] == '，':    # 第六个字符是逗号，则认为是五言绝句
                files_content += x
                count += 1
    # print("共有{}首五言律诗".format(count))
    
    # 字频统计
    words = sorted(list(files_content))  # 按汉字编码排序
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 低频字过滤
    delete_words = []
    for key in counted_words:
        if counted_words[key] <= 2:
            delete_words.append(key)
    for key in delete_words:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])   # 频数取负，即倒序

    words, _ = zip(*wordPairs)
    words += (" ",)

    # 构建 字到id的映射字典 与 id到字的映射字典
    word2idx = dict((c, i) for i, c in enumerate(words))
    idx2word = dict((i, c) for i, c in enumerate(words))
    word2idx_dic = lambda x: word2idx.get(x, len(words) - 1)
    return word2idx_dic, idx2word, words, files_content





class LSTMPoetryModel(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = True
        self.config = config

        # 诗歌训练文件预处理
        self.word2idx_dic, self.idx2word, self.words, self.files_content = preprocess_data(self.config)
        
        # 诗列表
        self.poems = self.files_content.split(']')
        # 诗的总数量
        self.poems_num = len(self.poems)
        
        # 如果有预训练好的模型文件，则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file) and self.loaded_model:
            self.model = load_model(self.config.weight_file)
        else:
            self.train()

    def build_model(self):
        '''LSTM模型构建'''
        print('模型构建中...')

        # 输入的维度
        input_tensor = Input(shape=(self.config.max_len, len(self.words)))
        # lstm层
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        # dropout层，丢弃50%，防止过拟合
        dropout = Dropout(0.5)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.5)(lstm)
        # dense层，也叫全连接层，将前面提取的特征，在dense经过非线性变化，提取这些特征之间的关联，最后映射到输出空间上
        dense = Dense(len(self.words), activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        # 优化器
        optimizer = Adam(lr=self.config.learning_rate)
        # 模型训练前的配置
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
    def sample(self, preds, temperature=1.0):
        '''
        temperature可以控制生成诗的创作自由约束度
        当temperature<1.0时，模型会做一些随机探索，输出相对比较新的内容
        当temperature>1.0时，模型预估方式偏保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds,1./temperature)
        preds = exp_preds / np.sum(exp_preds)
        prob = np.random.choice(range(len(preds)),1,p=preds)
        return int(prob.squeeze())
    
    def generate_sample_result(self, epoch, logs):
        '''训练过程中，每5个epoch打印出当前的学习情况'''
        if epoch % 5 != 0:
            return
        
        # 追加模式添加内容
        with open('out.txt', 'a',encoding='utf-8') as f:
            f.write('==================第{}轮=====================\n'.format(epoch))
                
        print("\n==================第{}轮=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------设定诗词创作自由度约束参数为{}--------------".format(diversity))
            generate = self.predict_random(temperature=diversity)
            print(generate)
            
            # 训练时的预测结果写入txt
            with open('out.txt', 'a',encoding='utf-8') as f:
                f.write(generate+'\n')
    
    def predict_random(self,temperature = 1):
        '''预估模式1：随机从库中选取一句开头的诗句，生成五言绝句'''
        if not self.model:
            print('没有预训练模型可用于加载！')
            return
        
        index = random.randint(0, self.poems_num)
        sentence = self.poems[index][: self.config.max_len]
        generate = self.predict_sen(sentence,temperature=temperature)
        return generate
    
    def predict_first(self, char,temperature =1):
        '''预估模式2：根据给出的首个字，生成五言绝句'''
        if not self.model:
            print('没有预训练模型可用于加载！')
            return
        
        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len个字+给出的首个文字作为初始输入
        sentence = self.poems[index][1-self.config.max_len:] + char
        generate = str(char)
        # 预测后面23个字
        generate += self._preds(sentence,length=23,temperature=temperature)
        return generate
    
    def predict_sen(self, text,temperature =1):
        '''预估模式3：根据给出的前max_len个字，生成诗句'''
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        if not self.model:
            return
        max_len = self.config.max_len
        if len(text)<max_len:
            print('给出的初始字数不低于 ',max_len)
            return

        sentence = text[-max_len:]
        # print('第一行为:',sentence)
        generate = str(sentence)
        generate += self._preds(sentence,length = 24-max_len,temperature=temperature)
        return generate
    
    def predict_hide(self, text,temperature = 1):
        '''预估模式4：根据给4个字，生成藏头诗五言绝句，例如输入'''
        if not self.model:
            print('没有预训练模型可用于加载！')
            return
        if len(text)!=4:
            print('藏头诗的输入必须是4个字！')
            return
        
        index = random.randint(0, self.poems_num)
        # 选取随机一首诗的最后max_len个字+给出的首个文字作为初始输入
        sentence = self.poems[index][1-self.config.max_len:] + text[0]
        generate = str(text[0])
        # print('第一行为 ',sentence)
        
        for i in range(5):
            next_char = self._pred(sentence,temperature)           
            sentence = sentence[1:] + next_char
            generate+= next_char
        
        for i in range(3):
            generate += text[i+1]
            sentence = sentence[1:] + text[i+1]
            for i in range(5):
                next_char = self._pred(sentence,temperature)           
                sentence = sentence[1:] + next_char
                generate+= next_char

        return generate
    
    
    def _preds(self,sentence,length = 23,temperature =1):
        '''
        供类内部调用的预估函数，输入max_len长度字符串，返回length长度的预测值字符串
        sentence:预测输入值
        lenth:预测出的字符串长度
        '''
        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence,temperature)
            generate += pred
            sentence = sentence[1:]+pred
        return generate
        
        
    def _pred(self,sentence,temperature =1):
        '''供类内部调用的预估函数，根据一串输入，返回单个预测字符'''
        if len(sentence) < self.config.max_len:
            print('in def _pred,length error ')
            return
        
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2idx_dic(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds,temperature=temperature)
        next_char = self.idx2word[next_index]
        
        return next_char

    def data_generator(self):
        '''生成器生成数据'''
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2idx_dic(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len, len(self.words)),
                dtype=np.bool
            )

            for t, char in enumerate(x):
                x_vec[0, t, self.word2idx_dic(char)] = 1.0

            yield x_vec, y_vec
            i += 1

    def train(self):
        '''训练模型'''
        print('开始训练...')
        number_of_epoch = len(self.files_content)-(self.config.max_len + 1)*self.poems_num
        number_of_epoch /= self.config.batch_size 
        number_of_epoch = int(number_of_epoch / 2.5)
        print('总迭代轮次为 ',number_of_epoch)
        print('总诗词数量为 ',self.poems_num)
        print('文件内容的长度为 ',len(self.files_content))
        # 构建模型
        if not self.model:
            self.build_model()
        # 模型训练
        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=50,
            callbacks=[
                ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )







