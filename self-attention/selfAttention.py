from keras import backend as K
from keras.datasets import imdb
from keras.engine.topology import  Layer

import pandas as pd
from keras_preprocessing import sequence
from tensorflow.python.keras import Input



class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)


    # train
max_features = 20000 #字典内的单词数/特征数
print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# IMDB 电影评论情感分类数据集
# 数据集来自 IMDB 的 25,000 条电影评论，以情绪（正面/负面）标记。
# 评论已经过预处理，并编码为词索引（整数）的序列表示。
# 为了方便起见，将词按数据集中出现的频率进行索引，例如整数 3 编码数据中第三个最频繁的词。
# 若num_words=20000:表示将数据限定为前20000个最常出现的单词，如果数据集中存在大于20000的单词，令其为2
# print('x_train',x_train) # x_train代表的一句话
# print('y_train',y_train) #情绪为正负面
# 标签转换为one-hot
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test) # get_dummies实现one-hot encode 正面/负面
# print('one-hot ytrain',y_train)
print(len(x_train), 'train sequences') #25000 train sequences
print(len(x_test),'test sequences') #25000 test sequences


# 数据归一化处理
maxlen = 64 #每句话的最大长度

print('pad sequences(sample x time)')
# 因为一句话长度参差不齐，所以用pad_sequences补齐，这样好处理数据
# pad_sequence 用0补足[[2,3,4]] -> [[0, 0, 0, 0, 0, 0, 0, 2, 3, 4]]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)


batch_size = 32
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import *
from keras.layers import Embedding,Attention,Dropout
from keras.layers import GlobalAveragePooling1D



S_inputs = Input(shape=(64,), dtype='int32')
# shape=(64,) 表示了预期的输入将是一批64维的向量
embeddings = Embedding(max_features,128)(S_inputs)    #max_features为字典内的大小，128位词向量的维度

O_seq = Self_Attention(128)(embeddings)
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)

outputs = Dense(2,activation='softmax')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)

print('model summary',model.summary())

# optimizer
opt = Adam(lr=0.0002,decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss,
              optimizer=opt,
              metrics=['accuracy'])
# compile将一个字符串编译为字节代码，metrics列表，包含评估模型在训练和测试时的性能指标
print('Train...')
h = model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=2,
              validation_data=(x_test,y_test))

# 画图
import matplotlib.pyplot as plt

plt.plot(h.history["loss"],label="train_loss")
plt.plot(h.history["val_loss"],label="val_loss")
plt.plot(h.history["accuracy"],label="train_acc")
plt.plot(h.history["val_accuracy"],label="val_acc")

plt.legend() #创建图例
plt.show()


