# 词向量 - Glove

---

GloVe：Global Vectors for Word Representation，它是一个基于**全局词频统计**的词表征工具。通过GloVe计算出的词向量捕捉到了词之间一些语义特性，比如相似性(similarity)、类比性(analogy)等。我们通过对向量的运算，比如欧氏距离或cos相似度，可以算出词之间的语义相似性。

##### 1、准备工作

由语料库构建一个共现矩阵(Co-occurrence Matrix) X，矩阵中的每一个元素![X_{ij}](https://math.jianshu.com/math?formula=X_%7Bij%7D)代表单词![i](https://math.jianshu.com/math?formula=i)和上下文单词![j](https://math.jianshu.com/math?formula=j)在特定大小的上下文窗口内共同出现的次数。这里GloVe做了额外的工作，它根据两个单词在上下文窗口的距离d，提出了一个衰减权重![decay = 1/d](https://math.jianshu.com/math?formula=decay%20%3D%201%2Fd)，也就是说距离越远的两个单词所占总计数的权重越小。

##### 2、使用GloVe模型训练词向量

原论文作者给出的loss function长这样：
 ![J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}](https://math.jianshu.com/math?formula=J%3D%5Csum_%7Bi%2C%20j%3D1%7D%5E%7BV%7D%20f%5Cleft(X_%7Bi%20j%7D%5Cright)%5Cleft(w_%7Bi%7D%5E%7BT%7D%20%5Ctilde%7Bw%7D_%7Bj%7D%2Bb_%7Bi%7D%2B%5Ctilde%7Bb%7D_%7Bj%7D-%5Clog%20X_%7Bi%20j%7D%5Cright)%5E%7B2%7D)
 其中![w_{i}^{T}](https://math.jianshu.com/math?formula=w_%7Bi%7D%5E%7BT%7D)和![\tilde{w}_{j}](https://math.jianshu.com/math?formula=%5Ctilde%7Bw%7D_%7Bj%7D)是最终求解的词向量，![b_{i}](https://math.jianshu.com/math?formula=b_%7Bi%7D)和![\tilde{b}_{j}](https://math.jianshu.com/math?formula=%5Ctilde%7Bb%7D_%7Bj%7D)是两个标量（作者定义的偏差），![f](https://math.jianshu.com/math?formula=f)是权重函数，V是词典的大小（共现矩阵的维度V x V）。这里我们知道GloVe模型没有使用神经网络做训练。

下面来说说权重函数![f](https://math.jianshu.com/math?formula=f)。在一个语料库中，存在很多单词他们在一起出现的次数是很多的，那么我们希望权重函数![f](https://math.jianshu.com/math?formula=f)满足以下几点：
 a. 这些单词的权重要大于那些很少在一起出现的单词，所以这个函数是非递减的；
 b. 我们不希望这个权重过大，当到达一定程度之后不再增加；
 c. 如果两个单词没有在一起出现，也就是![X_{ij} = 0](https://math.jianshu.com/math?formula=X_%7Bij%7D%20%3D%200)，那么它们应该不参与到loss function的计算中来，也就是![f(x)](https://math.jianshu.com/math?formula=f(x))要满足![f(0) = 0](https://math.jianshu.com/math?formula=f(0)%20%3D%200)。

综合以上，作者选择了如下的分段函数：



![img](https:////upload-images.jianshu.io/upload_images/1599787-5499a24f6be888a7.png?imageMogr2/auto-orient/strip|imageView2/2/w/714/format/webp)



作者在实验中取的![\alpha=0.75](https://math.jianshu.com/math?formula=%5Calpha%3D0.75)，![x_{max} = 100](https://math.jianshu.com/math?formula=x_%7Bmax%7D%20%3D%20100)。

这么看来GloVe像是一种无监督训练方法，因为不需要提供标注。但其实它还是有标注的，标注就是![\log X_{i j}](https://math.jianshu.com/math?formula=%5Clog%20X_%7Bi%20j%7D)，向量![w_{i}^{T}](https://math.jianshu.com/math?formula=w_%7Bi%7D%5E%7BT%7D)和![\tilde{w}_{j}](https://math.jianshu.com/math?formula=%5Ctilde%7Bw%7D_%7Bj%7D)是需要不断更新的学习参数，所以本质上还是监督学习，基于梯度下降。

> 按原论文中的说法：采用了AdaGrad的梯度下降算法，对矩阵![X](https://math.jianshu.com/math?formula=X)中的所有非零元素进行随机采样，学习速率设置为0.05，在vector size小于300的情况下迭代了50次，其他大小的verctors上迭代了100次，直至收敛。

最终学习得到的是两个向量![w_{i}^{T}](https://math.jianshu.com/math?formula=w_%7Bi%7D%5E%7BT%7D)和![\tilde{w}_{j}](https://math.jianshu.com/math?formula=%5Ctilde%7Bw%7D_%7Bj%7D)，因为![X](https://math.jianshu.com/math?formula=X)是对称的，所以从原理上![w_{i}^{T}](https://math.jianshu.com/math?formula=w_%7Bi%7D%5E%7BT%7D)和![\tilde{w}_{j}](https://math.jianshu.com/math?formula=%5Ctilde%7Bw%7D_%7Bj%7D)也是对称的，他们唯一的区别是初始化的值不一样，而导致最终的值不一样。所以这两者其实是等价的，都可以当成最终的结果来使用。但是为了提高鲁棒性，最终选择两者之和作为verctor（两者初始化不同，相当于加了不同的随机噪声，所以能提高鲁棒性）。

##### 3、作者如何构建模型的

上面我们了解了GloVe模型训练词向量的过程，但是我们还有一个很大的疑问，loss function是怎么来的，为什么是这个形式？

首先我们先熟悉几个符号定义：

- ![X_{ij}](https://math.jianshu.com/math?formula=X_%7Bij%7D)表示单词![j](https://math.jianshu.com/math?formula=j)出现在单词![i](https://math.jianshu.com/math?formula=i)的上下文中的次数；
- ![X_{i}=\sum_{j=1}^{N} X_{i, j}](https://math.jianshu.com/math?formula=X_%7Bi%7D%3D%5Csum_%7Bj%3D1%7D%5E%7BN%7D%20X_%7Bi%2C%20j%7D)， 表示单词![i](https://math.jianshu.com/math?formula=i)的上下文中所有单词出现的总次数；
- ![P_{i, k}=\frac{X_{i, k}}{X_{i}}](https://math.jianshu.com/math?formula=P_%7Bi%2C%20k%7D%3D%5Cfrac%7BX_%7Bi%2C%20k%7D%7D%7BX_%7Bi%7D%7D)， 表示单词![k](https://math.jianshu.com/math?formula=k)出现在单词![i](https://math.jianshu.com/math?formula=i)语境中的概率。
- ![ratio_{i, j, k}=\frac{P_{i, k}}{P_{j, k}}](https://math.jianshu.com/math?formula=ratio_%7Bi%2C%20j%2C%20k%7D%3D%5Cfrac%7BP_%7Bi%2C%20k%7D%7D%7BP_%7Bj%2C%20k%7D%7D)， 两个条件概率的比值。

作者从ratio中发现了一些规律，看下表：



![img](https:////upload-images.jianshu.io/upload_images/1599787-e77a4af8ef4ce84f.png?imageMogr2/auto-orient/strip|imageView2/2/w/966/format/webp)



看最后一行，我们可以用这个比值观察出两个单词![i](https://math.jianshu.com/math?formula=i)和![j](https://math.jianshu.com/math?formula=j)相对于单词![k](https://math.jianshu.com/math?formula=k)哪个更相关。

比如，ice和solid更相关，而stram和solid明显不相关，于是发现![\frac{P(k|ice)}{P(k|steam)}](https://math.jianshu.com/math?formula=%5Cfrac%7BP(k%7Cice)%7D%7BP(k%7Csteam)%7D)比1大很多；gas和steam更相关，而与ice不相关，所以![\frac{P(gas|ice)}{P(gas|steam)}](https://math.jianshu.com/math?formula=%5Cfrac%7BP(gas%7Cice)%7D%7BP(gas%7Csteam)%7D)远小于1；当都有关或都无关的时候，两者比例接近1.

所以，以上结论可以说明**通过概率的比例而不是概率本身去学习词向量可能是一个更恰当的方法**。

于是乎，作者开始了表演：

假设我们已经有了词向量![v_{i}、v_{j}、v_{k}](https://math.jianshu.com/math?formula=v_%7Bi%7D%E3%80%81v_%7Bj%7D%E3%80%81v_%7Bk%7D)，并且用这些词向量通过某种函数计算ratio能够得到同样的规律的话，就意味着我们词向量与共现矩阵具有很好的一致性，也就是说我们的词向量中蕴含了共现矩阵中蕴含的信息。翻译成数学表达式：`$\frac{P_{i, k}}{P_{j, k}}=r a t i o_{i, j, k}=g\left(v_{i}, v_{j}, v_{k}\right)$`，我们的目标应该是使得表达式两端尽可能地接近。所以，得出代价函数：
 `$J=\sum_{i, j, k}^{N}\left(\frac{P_{i, k}}{P_{j, k}}-g\left(v_{i}, v_{j}, v_{k}\right)\right)^{2}$`
 有个问题，模型中有3个单词，也就是说我们的计算复杂度是N x N x N，有点复杂。下面来做一些优化：

- 要考虑单词![i和j](https://math.jianshu.com/math?formula=i%E5%92%8Cj)之间的关系，那![g(v_{i}, v_{j}, v_{k})](https://math.jianshu.com/math?formula=g(v_%7Bi%7D%2C%20v_%7Bj%7D%2C%20v_%7Bk%7D))中大概要有![v_{j} - v_{j}](https://math.jianshu.com/math?formula=v_%7Bj%7D%20-%20v_%7Bj%7D)这么一项；
- ![ratio_{i,j,k}](https://math.jianshu.com/math?formula=ratio_%7Bi%2Cj%2Ck%7D)是个标量，那么![g(v_{i}, v_{j}, v_{k})](https://math.jianshu.com/math?formula=g(v_%7Bi%7D%2C%20v_%7Bj%7D%2C%20v_%7Bk%7D))最后应该是个标量，虽然输入都是向量，那内积应该是合理的，所以应该有这么一项:![\left(v_{i}-v_{j}\right)^{T} v_{k}](https://math.jianshu.com/math?formula=%5Cleft(v_%7Bi%7D-v_%7Bj%7D%5Cright)%5E%7BT%7D%20v_%7Bk%7D)；
- 然后作者又在![\left(v_{i}-v_{j}\right)^{T} v_{k}](https://math.jianshu.com/math?formula=%5Cleft(v_%7Bi%7D-v_%7Bj%7D%5Cright)%5E%7BT%7D%20v_%7Bk%7D)上加了一层指数运算，最终得到: 
- ![image-20200821092949330](https://note.youdao.com/yws/api/personal/file/WEBffabb7a646c693180fb2134b85c93ef3?method=getImage&version=3807&cstk=YgTyseBk)



下面把公式做进一步变形：
 ![image-20200821093013982](https://note.youdao.com/yws/api/personal/file/WEB0c7b5d9862059ee0d1b99b943816519b?method=getImage&version=3808&cstk=YgTyseBk)

因此，只要让分子分母对应相等即可：![P_{i, k}=\exp \left(v_{i}^{T} v_{k}\right)， P_{j, k}=\exp \left(v_{j}^{T} v_{k}\right)](https://math.jianshu.com/math?formula=P_%7Bi%2C%20k%7D%3D%5Cexp%20%5Cleft(v_%7Bi%7D%5E%7BT%7D%20v_%7Bk%7D%5Cright)%EF%BC%8C%20P_%7Bj%2C%20k%7D%3D%5Cexp%20%5Cleft(v_%7Bj%7D%5E%7BT%7D%20v_%7Bk%7D%5Cright))。因为分子分母形式相同，所以可以把两者统一考虑，我们有：![P_{i, j}=\exp \left(v_{i}^{T} v_{j}\right)](https://math.jianshu.com/math?formula=P_%7Bi%2C%20j%7D%3D%5Cexp%20%5Cleft(v_%7Bi%7D%5E%7BT%7D%20v_%7Bj%7D%5Cright))。

两边取对数：![\log \left(P_{i, j}\right)=v_{i}^{T} v_{j}](https://math.jianshu.com/math?formula=%5Clog%20%5Cleft(P_%7Bi%2C%20j%7D%5Cright)%3Dv_%7Bi%7D%5E%7BT%7D%20v_%7Bj%7D)

***loss function可以化简为：![image-20200821093041346](https://note.youdao.com/yws/api/personal/file/WEB2e87b92906a9ef7f929228f8dde02e3e?method=getImage&version=3809&cstk=YgTyseBk)

到这里，我们成功把计算复杂度从N x N x N降到N x N。

但是这里出现了新的问题，我们看两个式子：![\log \left(P_{i, j}\right)=v_{i}^{T} v_{j}](https://math.jianshu.com/math?formula=%5Clog%20%5Cleft(P_%7Bi%2C%20j%7D%5Cright)%3Dv_%7Bi%7D%5E%7BT%7D%20v_%7Bj%7D) 与 ![\log \left(P_{j, i}\right)=v_{j}^{T} v_{i}](https://math.jianshu.com/math?formula=%5Clog%20%5Cleft(P_%7Bj%2C%20i%7D%5Cright)%3Dv_%7Bj%7D%5E%7BT%7D%20v_%7Bi%7D)，两个等式的左边不等，但是右边却相等。

好了，继续脑洞。将代价函数中的条件概率展开：![\log \left(X_{i, j}\right)-\log \left(X_{i}\right)=v_{i}^{T} v_{j}](https://math.jianshu.com/math?formula=%5Clog%20%5Cleft(X_%7Bi%2C%20j%7D%5Cright)-%5Clog%20%5Cleft(X_%7Bi%7D%5Cright)%3Dv_%7Bi%7D%5E%7BT%7D%20v_%7Bj%7D)，添加偏置项并将![\log \left(X_{i}\right)](https://math.jianshu.com/math?formula=%5Clog%20%5Cleft(X_%7Bi%7D%5Cright))放到偏置里面去，所以有：
 ![image-20200821093217850](https://note.youdao.com/yws/api/personal/file/WEBd0a2f378148571ae7ad197edb8978fa1?method=getImage&version=3810&cstk=YgTyseBk)
 最后，出现频率越高的词对应的权重应该越大，所以在代价函数中添加权重项：
 ![image-20200821093402306](https://note.youdao.com/yws/api/personal/file/WEBe295e2884d601ea2ef5464758f6bf9ba?method=getImage&version=3812&cstk=YgTyseBk)

> 参考：
>
> [https://www.aclweb.org/anthology/D14-1162](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.aclweb.org%2Fanthology%2FD14-1162)
>
> [https://blog.csdn.net/coderTC/article/details/73864097](https://links.jianshu.com/go?to=https%3A%2F%2Fblog.csdn.net%2FcoderTC%2Farticle%2Fdetails%2F73864097)



