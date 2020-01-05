from flask import Flask, render_template, request
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import json
import jieba  # 结巴分词
import re
import warnings
from gensim.models import KeyedVectors
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import os

app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True

'''
加载模型和预测模块
'''
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# 程序开始时声明
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

warnings.filterwarnings("ignore")
# 使用gensim加载预训练中文分词embedding
corpus_path = r'D:\PycharmProjects\python_homework\model/'
cn_model = KeyedVectors.load_word2vec_format(corpus_path + 'sgns.zhihu.bigram', binary=False)
from tensorflow.python.keras.models import load_model
# 在model加载前添加set_session
set_session(sess)
model = load_model('D:\\PycharmProjects\\python_homework\\model_weight.h5')
emos = ['anger', 'disgust', 'happiness', 'like', 'sadness']
if os.path.exists('D:\\stanford_nlp\\stanford-corenlp-full-2018-10-05'):
    print("corenlp exists")
else:
    print("corenlp not exists")
nlp = StanfordCoreNLP('D:\\stanford_nlp\\stanford-corenlp-full-2018-10-05', lang='zh')

preInfo = {}
@app.route('/cont_post', methods=['GET', 'POST'])  # 路由
def cont_post():
    preInfo2 = {}
    cont = request.get_data()
    cont =cont.decode('utf-8')
    print(cont)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        preInfo['emo'], preInfo['conf'], preInfo['s_char'], preInfo['texts'] = predict_sentiment(str(cont))
        for i in range(len(preInfo['emo'])):
            preInfo2[i] = [preInfo['s_char'][i],preInfo['emo'][i],preInfo['conf'][i],preInfo['texts'][i]]
    print(preInfo)
    print(preInfo2)
    return json.dumps(preInfo2)

#预测情感
def predict_sentiment(te):
    print(te)
    texts = te.replace('！', '。').replace('“', '').replace('”', '').replace('\n\n','。').split('。')
    print('texts',texts)

    pre_emo = []
    pre_conf = []
    pre_char =[]
    for text in texts:
        if text == '':
            continue
        pre_char.append(subj(text))
        # 去标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        # 分词
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        # tokenize
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0
        # padding
        tokens_pad = pad_sequences([cut_list], maxlen=38,
                                   padding='pre', truncating='pre')
        print('ok================================ok')
        # 预测
        result = model.predict(x=tokens_pad)
        ind = np.argmax(result[0])
        p = np.max(result[0])*100
        pre_emo.append(emos[ind])
        pre_conf.append(format(p,'.2f'))
        print('预测得到的情感为：',emos[ind],'可靠性为：',format(p,'.2f'),'%')
    return pre_emo,pre_conf,pre_char,texts

#提取主语
def subj(sentence):

    # sentence = '王明是清华大学的一个研究生'
    # sentence = '廊坊是个好地方，紧邻京津，有点闹中取静的意思'
    # print (nlp.word_tokenize(sentence))   #分词
    # print (nlp.pos_tag(sentence))     #词性
    # print(nlp.ner(sentence))  # NER
    #
    # print (nlp.parse(sentence))     #语法分析
    # print (nlp.dependency_parse(sentence))   #语法依赖关系
    res = nlp.dependency_parse(sentence)
    sub = []
    for i in range(len(res)):
        if res[i][0] == 'nsubj':
            sub.append(res[i][2])
            break
    tmp = nlp.word_tokenize(sentence)
    try:
        re =  tmp[sub[0]-1]
    except Exception as e:
        re= "NULL"
    return re

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/keke')
# def keke():
#     return render_template('keke.html')

if __name__ == '__main__':
    app.run()
