# STAT 157学习过程记录

用于记录完成STAT 157的学习过程

---

2019/7/29

- [Logistics, Software, Linear Algebra](https://courses.d2l.ai/berkeley-stat-157/units/introduction.html)

  - 主要介绍相关软件以及线性代数的基础知识，由于我使用的设备不支持GPU，所以不考虑使用推荐的软件，关于线性代数的作业也比较基础，并且之前在其它课程中对向量、矩阵等运算已经有了不少训练，因此这一部分暂且略过

- [Probability and Statistics (Bayes Rule, Sampling Naive Bayes, Sampling)](https://courses.d2l.ai/berkeley-stat-157/units/probability.html)

  - 朴素贝叶斯：课程中以垃圾邮件分类为例大致讲解了朴素贝叶斯的原理，但是非常简略，以至于我没有听懂。。。虽然之前已经学过朴素贝叶斯，但是很久没有接触导致忘的差不多了，下面是根据一篇[教程](https://www.cnblogs.com/leoo2sk/archive/2010/09/17/naive-bayesian-classifier.html)对朴素贝叶斯的一个回顾

    - 假设待分类项$x$具有特征$a_1, a_2, \cdots, a_n$，（例如对于邮件而言可以把单词是否出现作为特征），$x$可能的分类有$y_1, y_2,\cdots, y_m$，我们需要对给定的$x$确定最优分类$y_{max}$

    - 如何利用朴素贝叶斯的原理来解决上面的问题呢？

      - 首先看一下条件概率的计算
        $$
        \begin{align}
        P(A|B)&=\frac{P(AB)}{P(B)}\\
        &=\frac{P(B|A)P(A)}{P(B)}
        \end{align}
        $$
        将$B$看作是特征，$A$看作分类，我们要求的是$P(y=y_i|x=(a_1, a_2, \cdots, a_n))$，因为对于同一个分类项$x$来说，$P(x=(a_1, a_2, \cdots, a_n))$是一个常数，因此在计算不同可能分类的条件概率时我们不用考虑它，也就是说$P(y=y_i|x=(a_1, a_2, \cdots, a_n))\propto P(x=(a_1, a_2, \cdots, a_n)|y=y_i)\cdot P(y_i)$

        当认为各个特征选择完全独立时，$P(x=(a_1, a_2, \cdots, a_n)|y=y_i)=P(x_1=a_1|y=y_i)\cdots  P(x_n=a_n|y=y_i)$

        即：$P(y=y_i|x=(a_1, a_2, \cdots, a_n))\propto P(x_1=a_1|y=y_i)\cdots  P(x_n=a_n|y=y_i)\cdot P(y_i)$，也是课程中的说明。

      - 有了上面的简要介绍，接下来我们可以针对具体问题来利用朴素贝叶斯来构建分类模型了。与课程一样，这里以MNIST手写数字为例，在MNIST数据集中，$x$可以看作一张图片的所有像素点，每个像素点的取值选择为0/1，$y$总共有十种，从0-9。由此，我们基本对问题定义有了大概的了解，接着需要利用训练集计算$P(x_i=1|y_j)$以及$P(y=y_j)$其中$ i=1～784, j=0～9$，$i$表示像素点的序号，对于测试样例，只要计算$P(x_1=a_1|y=y_j)\cdots  P(x_{784}=a_n|y=y_j)\cdot P(y_j)$取得最大值的$j$即为预测结果。

      - 具体的分类器构建可以参考[STAT 156 Naive Bayes](https://courses.d2l.ai/berkeley-stat-157/slides/1_24/naive-bayes.ipynb)，为了加深对朴素贝叶斯的理解，我自己也构建了一个分类器，[Naive Bayes Classifier](https://github.com/waxin/STAT-157/blob/master/NaiveBayesClassifier.ipynb)

        - 在自己实现分类器的过程中，我认为有两点需要注意，其一是$P(x_i=1|y_j)$计算时分母是$y_i$对应的统计数目而非整个训练集大小，其二可以认为是一个小track，在某些情况下$P(x_i=1|y_j)$可能为0，因为需要进行$log$操作，所以这时会报`warning`，为了避免，可以按照下面的方式初始化：

          ```python
          xcount = np.ones((10, 28*28))
          ycount = np.ones((1, 10))
          ```

          