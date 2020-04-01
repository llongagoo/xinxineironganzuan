from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import *
import os, os.path, shutil
import string

# 制作停止词表
def stopword():
    stopword = []
    f = open('stopword.txt', 'r')
    for i in f:
        i = i.strip()
        stopword.append(i)
    return stopword

# 获取词袋
def get_bag(mail_list):
    word_bag = []
    for mail in mail_list:
        word_list = mail.split()
        for i in word_list:
            if len(i)>1:
                if i not in word_bag:
                    if i not in stopword():
                        word_bag.append(i)

    return word_bag

# 获取邮件列表
def get_mail():
    mail_list = []
    label = []
    txtPath = 'email/ham/'
    txtLists = os.listdir(txtPath)  # 列出文件夹下所有的目录与文件
    txtPath2 = 'email/spam/'
    txtLists2 = os.listdir(txtPath2)  # 列出文件夹下所有的目录与文件
    for txt in txtLists:
        try:
            f = open(txtPath + txt, 'r', encoding='utf-8')  # 返回一个文件对象
            mail = f.read()  # 读取全部内容
            for c in string.punctuation:
                mail = mail.replace(c, '')
            mail_list.append(mail)
            label.append(0)
        except:
            continue
    for txt in txtLists2:
        try:
            f = open(txtPath2 + txt, 'r', encoding='utf-8')  # 返回一个文件对象
            mail = f.read()  # 读取全部内容
            for c in string.punctuation:
                 mail = mail.replace(c, '')
            mail_list.append(mail)
            label.append(1)
        except:
            continue
    return mail_list,label

# 获取所有bool向量
def get_bool(mail_list,word_bag,label):
    word_bool = []
    for mail in mail_list:
        mail_bool = []
        word_list = mail.split()
        for i in word_bag:
            if i in word_list:
                mail_bool.append(1)
            else:
                mail_bool.append(0)
        word_bool.append(mail_bool)
    return word_bool,label


# 获取所有数量向量
def get_num(mail_list,word_bag,label):
    word_num = []
    for mail in mail_list:
        mail_num = []
        word_list = mail.split()
        for i in word_bag:
            if i in word_list:
                mail_num.append(word_list.count(i))
            else:
                mail_num.append(0)
        word_num.append(mail_num)
    return word_num,label

def improt_data_with_num():
    mail_list, label = get_mail()
    word_bag = get_bag(mail_list)
    return get_num(mail_list, word_bag, label)

def improt_data_with_bool():
    mail_list, label = get_mail()
    word_bag = get_bag(mail_list)
    return get_bool(mail_list, word_bag, label)

#if __name__ == '__main__':
#    mail_list,label = get_mail()
#    word_bag = get_bag(mail_list)
#    print(get_num(mail_list,word_bag,label))
#    print(get_bool(mail_list,word_bag,label))



def trainNB0(trainMatrix,trainCategory):#文档矩阵和文档类型标签
    numtTrainDocs=len(trainMatrix)#一共有几个档
    print(numtTrainDocs)
    numWordS=len(trainMatrix[0])#这个文档有几个词
    print(numWordS)
    pAbusive=sum(trainCategory)/float(numtTrainDocs)#垃圾邮箱总占比
    print(pAbusive)
    p0Num=ones(numWordS)#返回一个用1填充的数组
    p1Num=ones(numWordS)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numtTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect =log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def train_byes(x_train,y_train):
    '''
    :param x_train:
    :param y_train:
    :return:
        P_c1    垃圾邮件的概率
        p_w     p(w0)*p(w1)*p(w2*)...*p(wn)
        p_w_c1: p(w0|c1)*p(w1|c1)*p(w2|c1)...*p(wn|c1)
    '''
    train_sum = len(y_train)
    num_word = len(x_train[0])
    p_c1 = sum(y_train)/float(train_sum)#垃圾邮件的概率
    pw_times=np.ones(num_word)
    pwc1_times=np.ones(num_word)

def train_bayes(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def main_bayes(x_train, x_test, y_train, y_test):
    # 0代表正常 1代表垃圾
    p0V, p1V, pClass1 = train_bayes(array(x_train), array(y_train))
    y_predict=[]
    for test in x_test:
        y_predict.append(classifyNB(test, p0V, p1V, pClass1))
    print(classification_report(y_test, y_predict))

if __name__ == '__main__':
    all,label=improt_data_with_bool()
    x_train, x_test, y_train, y_test = train_test_split(all, label, test_size=0.2)
    print("朴素贝叶斯的精确度，召回率：")
    main_bayes(x_train, x_test, y_train, y_test)

