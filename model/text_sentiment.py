import numpy as np
import matplotlib.pyplot as plt
import re
import jieba  # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
import glob

warnings.filterwarnings("ignore")
# 使用gensim加载预训练中文分词embedding
corpus_path = r'D:\PycharmProjects\python_homework\model/'
cn_model = KeyedVectors.load_word2vec_format(corpus_path + 'sgns.zhihu.bigram', binary=False)

'''
将所有的情感句子合并到train_texts_orig（为list）中
train_texts_orig ：所有情感的句子的list
nums ：记录每类情感第一个句子在list中索引
emos ：情感名字list
len_emos：每类情感的个数
添加完所有样本之后，train_texts_orig为一个含有14306条文本的list
emos=['anger', 'disgus', 'happiness', 'like', 'sadness'],
nums=[0, 1860, 4933, 7805, 11911, 14306]
'''


# 读取txt文本内容的函数
def read_content(file):
    content = []
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            content.append(line.split('\t')[1].strip('\n'))
    return content

path = (r'D:\PycharmProjects\python_homework\model\dataset')
train_texts_orig = []
nums = [0]
emos = []
len_emos = []
for em_txts in glob.glob(path + '/*.txt'):
    emo = em_txts.split('\\')[-1].strip('.txt')
    emos.append(emo)
    tmp_emotion = read_content(em_txts)
    len_emos.append(len(tmp_emotion))
    # print(emo,'的样本数有:',len(tmp_emotion))
    train_texts_orig = train_texts_orig + tmp_emotion
    nums.append(len(train_texts_orig))

print('样本总共: ' + str(nums[-1]))
# print(train_texts_orig[:10])

'''对数据分布进行柱状图的展示'''
# import plotly as py
# import plotly.graph_objs as go
# pyplt = py.offline.plot
# # Trace
# trace_basic = [go.Bar(
# x = emos,
# y = len_emos,
# )]
# # Layout
# layout_basic = go.Layout(title = 'Data',xaxis = go.XAxis(range = [-0.5,4.5], domain = [0,1]))
# # Figure
# figure_basic = go.Figure(data = trace_basic, layout = layout_basic)
# # Plot
# pyplt(figure_basic, filename='/1.html')

# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional, CuDNNLSTM
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

'''
******分词和tokenize*******
首先我们去掉每个样本的标点符号，然后用jieba分词，jieba分词返回一个生成器，没法直接进行tokenize，
所以我们将分词结果转换成一个list，并将它索引化，这样每一例评价的文本变成一段索引数字，对应着预训练词向量模型中的词。
进行分词和tokenize
train_tokens是一个长长的list，其中含有14306个小list，对应每一条评价
'''
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

'''
**索引长度标准化**
因为每段文本的长度是不一样的，如果单纯取最长的一个文本，
并把其他文本填充成同样的长度，这样十分浪费计算资源，所以取一个折衷的长度。
'''

# 获得所有tokens的长度
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)

# 平均tokens的长度
print('平均tokens的长度：', np.mean(num_tokens))

# plt.hist(np.log(num_tokens), bins = 100)  # 服从正态分布
plt.hist(num_tokens, bins=100)
plt.xlim((0, 100))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 取tokens平均值并加上两个tokens的标准差，
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print('max_tokens:', max_tokens)

'''
**准备Embedding Matrix**
为模型准备embedding matrix（词向量矩阵），根据keras的要求，
需要准备一个维度为$(numwords, embeddingdim)$的矩阵，
num words代表使用的词汇的数量，emdedding dimension在现在使用的预训练词向量模型中是300，
每一个词汇都用一个长度为300的向量表示。  
注意只选择使用前50k个使用频率最高的词，在这个预训练词向量模型中，一共有260万词汇量，
如果全部使用在分类问题上会很浪费计算资源，因为的训练样本很小，一共只有14k，
如果有100k，200k甚至更多的训练样本时，在分类问题上可以考虑减少使用的词汇量。
'''

# 词向量的size
embedding_dim = cn_model.vector_size
# 词典共有259883个词，也可以取前50000
num_words = 259883
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 259883 * 300
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')

# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
print('embedding_matrix.shape:', embedding_matrix.shape)

'''
**padding（填充）和truncating（修剪）**  
把文本转换为tokens（索引）之后，每一串索引的长度并不相等，所以为了方便模型的训练需要把索引的长度标准化，
上面选择了38这个可以涵盖95%训练样本的长度，接下来进行padding和truncating，
一般采用'pre'的方法，这会在文本索引的前面填充0，因为根据一些研究资料中的实践，如果在文本索引后面填充0的话，会对模型造成一些不良影响。

'''
# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                          padding='pre', truncating='pre')

# 超出25万个词向量的词用0代替
train_pad[train_pad >= num_words] = 0

# 可见padding之后前面的tokens全变成0，文本在最后面
print('train_pad[33]', train_pad[33]) #随机选取

# 准备target向量，[1860, 3073, 2872, 4106, 2395]，anger：0.，disgus：1.，happiness：2.，like：3.，sadness：4.
train_target = np.concatenate((np.zeros(1860), np.ones(3073), np.ones(2872)*2, np.ones(4106)*3, np.ones(2395)*4))

# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split

# 90%的样本用来训练，剩余10%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)

# 用LSTM对样本进行分类
model = Sequential()

# 模型第一层为embedding
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))

# model.add(Bidirectional(CuDNNLSTM(units=32, return_sequences=True)))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
# model.add(CuDNNLSTM(units=16, return_sequences=False))
model.add(LSTM(units=16, return_sequences=False))

model.add(Dense(5, activation='softmax'))
# 使用adam以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

# 开始训练
model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=50,
          batch_size=128)
# callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
# **结论**
# 首先对测试样本进行预测，得到了还算满意的准确度。
# 之后定义一个预测函数，来预测输入的文本的极性，可见模型对于否定句和一些简单的逻辑结构都可以进行准确的判断。

result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))

model.save('model_weight.h5')
'''
调用模型进行情感分类测试
'''
# from tensorflow.python.keras.models import load_model
# model = load_model('D:\\PycharmProjects\\python_homework\\model_weight.h5')

# def predict_sentiment(text):
#     print(text)
#     # 去标点
#     text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
#     # 分词
#     cut = jieba.cut(text)
#     cut_list = [i for i in cut]
#     # tokenize
#     for i, word in enumerate(cut_list):
#         try:
#             cut_list[i] = cn_model.vocab[word].index
#         except KeyError:
#             cut_list[i] = 0
#     # padding
#     tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
#                                padding='pre', truncating='pre')
#     # 预测
#     result = model.predict(x=tokens_pad)
#     ind = np.argmax(result[0])
#     p = np.max(result[0])*100
#     print('预测得到的情感为：',emos[ind],'可靠性为：',format(p,'.2f'),'%')
    # if coef >= 0.5:
    #     print('是一例正面评价', 'output=%.2f' % coef)
    # else:
    #     print('是一例负面评价', 'output=%.2f' % coef)


# test_list = [
#
#     '最美的天空来自自由的阳光'
# ]
# for text in test_list:
#     predict_sentiment(text)
