# 词向量 - FastText

---

- [词向量学习笔记（三）FastText](https://blog.csdn.net/weixin_43886056/article/details/107889985#FastText_1)

- - [一、概述](https://blog.csdn.net/weixin_43886056/article/details/107889985#_4)

  - [二、FastText模型原理](https://blog.csdn.net/weixin_43886056/article/details/107889985#FastText_14)

  - - [2.1 FastText架构](https://blog.csdn.net/weixin_43886056/article/details/107889985#21_FastText_16)
    - [2.1 Softmax回归和层次Softmax](https://blog.csdn.net/weixin_43886056/article/details/107889985#21_SoftmaxSoftmax_23)
    - [2.2 n-gram特征](https://blog.csdn.net/weixin_43886056/article/details/107889985#22_ngram_66)

  - [三、源代码分析](https://blog.csdn.net/weixin_43886056/article/details/107889985#_70)

  - - [3.1 总体结构](https://blog.csdn.net/weixin_43886056/article/details/107889985#31__74)
    - [3.2 训练格式](https://blog.csdn.net/weixin_43886056/article/details/107889985#32__79)
    - [3.3 具体模块实现](https://blog.csdn.net/weixin_43886056/article/details/107889985#33__85)

  - [四、FastText的Python应用](https://blog.csdn.net/weixin_43886056/article/details/107889985#FastTextPython_99)

  - - [4.1 模型的训练](https://blog.csdn.net/weixin_43886056/article/details/107889985#41__101)

    - [4.2 模型的应用](https://blog.csdn.net/weixin_43886056/article/details/107889985#42__137)

    - - [4.2.0 查看一个词的词向量](https://blog.csdn.net/weixin_43886056/article/details/107889985#420__139)
      - [4.2.1 计算和一个词语最相关的词](https://blog.csdn.net/weixin_43886056/article/details/107889985#421__149)
      - [4.2.2 计算两个词的相似度](https://blog.csdn.net/weixin_43886056/article/details/107889985#422__162)
      - [4.2.3 查找不同类的词](https://blog.csdn.net/weixin_43886056/article/details/107889985#423__175)
      - [4.2.4 查看对应关系](https://blog.csdn.net/weixin_43886056/article/details/107889985#424__188)
      - [4.2.5 计算单词移动距离WMD](https://blog.csdn.net/weixin_43886056/article/details/107889985#425_WMD_200)

  - [五、总结](https://blog.csdn.net/weixin_43886056/article/details/107889985#_220)

  - - [5.1 FastText的特性](https://blog.csdn.net/weixin_43886056/article/details/107889985#51_FastText_222)
    - [5.2 FastText和Word2Vec的区别](https://blog.csdn.net/weixin_43886056/article/details/107889985#52_FastTextWord2Vec_228)



## 一、概述

  FastText是Facebook于2016年开源的一个词向量计算和文本分类工具，提供简单而高效的文本分类和表征学习的方法，典型应用场景是“带监督的文本分类问题”。在文本分类任务中，FastText（浅层网络）往往能取得和深度网络相媲美的精度，却在训练时间上比深度网络快许多数量级。在标准的多核CPU上，能够训练10亿词级别语料库的词向量在10分钟之内，能够分类有着30万多类别的50多万句子在1分钟之内。
  FastText结合了自然语言处理和机器学习中最成功的理念。这些包括了使用词袋以及n-gram袋表征语句，还有使用子词(subword)信息，并通过隐藏表征在类别间共享信息。另外采用了一个softmax层级(利用了类别不均衡分布的优势)来加速运算过程。
  FastText与基于神经网络的分类算法相比有如下优点：

- FastText在保持高精度的情况下加快了训练速度和测试速度
- FastText不需要预训练好的词向量，FastText会自己训练词向量
- FastText两个重要的优化：Hierarchical Softmax、N-gram

## 二、FastText模型原理

### 2.1 FastText架构

  FastText模型架构和word2vec中的CBOW很相似，不同之处是FastText预测标签，而CBOW预测的是中间词，即模型架构类似但是模型的任务不同。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094210850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

  如上图所示，其中`$x1,x2,…,xN−1,xNx_{1},x_{2},…,x_{N−1},x_{N}*x*1,*x*2,…,*x**N*−1,*x**N*$`表示一个文本中的n-gram向量，每个特征是词向量的平均值。此处用全部的n-gram去预测指定类别。

### 2.1 Softmax回归和层次Softmax

  Softmax回归（Softmax Regression）又被称作多项逻辑回归（multinomial logistic regression），它是逻辑回归在处理多类别任务上的推广。
。

  **分层softmax**
  对于有大量类别的数据集，fastText使用了一个分层分类器（而非扁平式架构）。不同的类别被整合进树形结构中。层次Softmax建立在哈弗曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。
  标准的Softmax回归中，要计算y=j时的Softmax概率P(y=j)P(y=j)*P*(*y*=*j*)需要对所有的K个概率做归一化，这在∣y∣|y|∣*y*∣很大时非常耗时。而分层Softmax的基本思想是使用树的层级结构替代扁平化的标准Softmax，使得在计算P(y=j)P(y=j)*P*(*y*=*j*)时，只需计算一条路径上的所有节点的概率值，无需在意其它的节点。
  如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809093927395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)
  树的结构是根据类标的频数构造的霍夫曼树。K个不同的类标组成所有的叶子节点，K-1个内部节点作为内部参数，从根节点到某个叶子节点经过的节点和边形成一条路径，路径长度被表示为`$L(yi)L(y_i)*L*(*y**i*​)$`
  此外，通过使用Huffman算法建立用于表征类别的树形结构，频繁出现类别的树形结构的深度要比不频繁出现类别的树形结构的深度要小，这也使得进一步的计算效率更高。

### 2.2 n-gram特征

  n-gram是一种基于语言模型的文本特征提取算法，基本思想是将文本内容按照字节顺序进行大小为N的滑动窗口操作，最终形成长度为N的字节片段序列。为了区分前后缀情况，"<"， ">"符号被加到了词的前后端。除了词的子串外，词本身也被包含进了 n-gram字母串包。以where为例，n=3 的情况下，其子串分别为<wh, whe, her, ere, re>，以及其本身。

## 三、源代码分析

  FaceBook开源的fasttext代码地址：https://github.com/facebookresearch/fastText。

### 3.1 总体结构

  FastText的代码结构和各模块的功能及其之间的联系如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020080909394733.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

### 3.2 训练格式

- 一行一个句子，每个词用空格分割
- 如果一个词带有前缀“**label**”，那么它就作为一个类标签
- 训练文件支持UTF-8格式。

### 3.3 具体模块实现

- fasttext

  ：最顶层的模块，它的主要功能是训练和预测。

  - FastText::train：初始化词典、输入输出层等参数，利用C++标准库的thread实现多线程训练
  - FastText::trainThread：实现了标准的随机梯度下降，并随着训练的进行逐步降低学习率。
  - FastText::supervised/cbow/skipgram：不同的模型更新策略，由trainThread调用，更新模型
  - FastText::predict：模型预测model.predict的上层封装，获取测试数据集的分类和预测

- model

  ：model模块对外提供的服务可以分为update和predict两类。

  - Model::update：该函数有三个参数，分别是“输入”，“类标签”，“学习率”。通过前向传播，反向传播（负采样/层次softmax/普通softmax）等过程对模型进行更新
  - Model::predict：用于给输入数据打上 1~K个类标签，并输出各个类标签对应的概率值。

- 其他模块

  ：

  - dictionary：完成了训练文件载入，哈希表构建，word n-gram计算等。
  - Matrix, Vector：封装了简单的矩阵向量操作

## 四、FastText的Python应用

### 4.1 模型的训练

- **用法**：gensim.models.fasttext(args)

- 常规参数：

  - min_count=5: 忽略稀疏词下界
  - size=100: 词嵌入大小
  - alpha=0.025: 初始学习率
  - window=5: 上下文窗口
  - workers=3: 用于控制训练的并行数
  - sg=0: 模型的选择: 0是CBOW模型，1则是Skip-Gram模型。
  - hs=0: 0是负采样，1是层次Softmax。
  - sample=1e-3: 下采样高频词的阈值
  - negative=5: 需要采样的noise words
  - iter=5: 语料库上的迭代数
  - sorted_vocab=1: 为1表示按频率降序对词表进行排序

- 特殊参数

  ：

  - min_n=3：字符的最小长度
  - max_n=6：字符的最大长度
  - bucket=2000000：hash存储时模型使用的桶数
  - word_ngrams=1: 如果为1，则使用子词(n个符号)信息来丰富词向量。

```python
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
# 训练模型
sentences = LineSentence(r'E:/Machine Learning/Demo/wiki.zh.word.txt')
model = FastText(sentences, size=150, window=5, min_count=5, iter=10, min_n=3, max_n=6, word_ngrams=0,workers=4)
# 保存模型
model.save('fasttext_150')
# 加载模型
model = FastText.load('fasttext_150')
# 更新模型
model.build_vocab(sentences_new, update=True)
model.train(sentences_new, total_examples=model.corpus_count, epochs=model.iter)
123456789101112
```

### 4.2 模型的应用

#### 4.2.0 查看一个词的词向量

- **用法**：model[vocabulary : str]/model.wv[vocabulary : str]

```python
print(model['机器学习'])  # 词向量获得的方式
print(model.wv['机器学习'])  # 词向量获得的方式
12
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020080909400630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

#### 4.2.1 计算和一个词语最相关的词

- **用法**：model.most_similar()

```python
items = model.most_similar('机器学习', topn=8)
print("与'机器学习'相关的词有：")
for item in items:
    # 词的内容，词的相关度
    print(item[0], item[1])
12345
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094017777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

#### 4.2.2 计算两个词的相似度

- **用法**：model.similarity()

```python
# 计算两个词语之间的相关度
val1 = model.similarity('机器学习', '深度学习')
val2 = model.similarity('国王', '皇帝')
print("机器学习/深度学习的相似度为：", val1)
print("国王/皇帝的相似度为：", val2)
12345
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094028147.png)

#### 4.2.3 查找不同类的词

- **用法**：model.doesnt_match()

```python
# 找出不同类的词
word = model.doesnt_match(['总统', '国王', '主席', '酋长', '司令'])
print("在【'总统', '国王', '主席', '酋长', '司令'】中不同类的词为：", word)
word = model.doesnt_match(['机器学习', '深度学习', '强化学习', '好好学习'])
print("在【'机器学习', '深度学习', '强化学习', '好好学习'】中不同类的词为：", word)
12345
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094037159.png)

#### 4.2.4 查看对应关系

- **用法**：model.most_similar()/model.wv.most_similar_cosmul()

```python
# 寻找对应关系
value = model.most_similar(positive=['超级计算机', '便携'], negative=['高效'], topn=8)
print("'超级计算机'->'高效'，则'？'->'便携':")
print(np.array(value))
1234
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094048452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

#### 4.2.5 计算单词移动距离WMD

- **用法**：model.wmdistance(sentence1, sentence2)

```python
mytext1 = "历史从来都是即兴之作。而当它成为历史，才被千秋万代喋喋不休地评论。"
mytext2 = "。历史是人的足迹。但并不是所有留下足迹的人都敢于正视自己的历史。历史是无法重写的。"
print(mytext1,'\n',mytext2)
sentence1 = jieba.lcut(mytext1,cut_all=False)
sentence2 = jieba.lcut(mytext2,cut_all=False)
# print(sentence1,'\n',sentence2)
stopwords = stopwords.words('chinese')
sentence1 = [_ for _ in sentence1 if _ not in stopwords]
sentence2 = [_ for _ in sentence2 if _ not in stopwords]
print(sentence1,'\n',sentence2)
distance = model.wmdistance(sentence1, sentence2)
print('二者的移动距离为：',distance)
123456789101112
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200809094057131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg4NjA1Ng==,size_16,color_FFFFFF,t_70)

## 五、总结

### 5.1 FastText的特性

  FastText适用与分类类别非常大而且数据集足够多的情况，当分类类别比较小或者数据集比较少的话，很容易过拟合。
  FastText可以完成无监督的词向量的学习，学习出来词向量来保持住词和词之间，相关词之间是一个距离比较近的情况；也可以用于有监督学习的文本分类任务。
  FastText相当于Word2Vec中cbow和h-softmax的灵活使用。

### 5.2 FastText和Word2Vec的区别

- 相似处

  ：

  - 图模型结构都采用embedding向量得到word的隐向量表达。
  - 都采用如使用Hierarchical softmax优化训练等优化方式

- 不同处

  ：

  - 模型的输出层：Word2Vec的输出层对应的是每一个term；而FastText的输出层是分类的label。
  - 模型的输入层：Word2Vec的输入层是上下文窗口内的term；而FastText是整个sentence的内容。
  - 训练目的：Word2Vec的目的是得到词向量，FastText则利用了h-softmax的分类功能，找到概率最大的label。