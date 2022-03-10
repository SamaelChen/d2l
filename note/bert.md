---
presentation:
    theme: blood.css
    width: 1600
    height: 900
---

<!-- slide -->
# 预训练模型
**from word2vec to bert**

<!-- slide -->
<p align='left'><font size="20px">One-Hot </font></p>
<frameset cols="25%,50%,25%">
<div style='width:50%;float:left;height:700px;'>
<p align='left'><font size="6px">为了得到索引为$i$的任意词的one-hot向量表示，我们创建了一个全为0的长度为$N$的向量，并将位置$i$的元素设置为1。这样，每个词都被表示为一个长度为$N$的向量，可以直接由神经网络使用。</font></p>

<p align='left'><font size="6px">由于任意两个不同词的one-hot向量之间的余弦相似度为0，所以one-hot向量不能编码词之间的相似性。</font></p>

<p>$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$</p>
</div>
<div style='width:50%;float:left;height:700px;'>
$$
W = 
\begin{bmatrix}
1      & \cdots & \cdots & \cdots & 0      \\
\vdots & \cdots & \vdots & \cdots & \vdots \\
0      & \cdots & 1      & \cdots & 0      \\
\vdots & \cdots & \vdots & \cdots & \vdots \\
0      & \cdots & \cdots & \cdots & 1
\end{bmatrix}
$$
</div>

<!-- slide -->
<p align='left'><font size="20px">Window-based Co-occurence Matrix</font></p>
<div style='width:50%;float:left;height:600px;'><p align='left'><font size="6px">建立起窗口大小为1的共生矩阵，矩阵中数字代表两个单词出现在同一个窗口的频率，通过建立共生矩阵就可以很好的计算两个单词的相似性了。</font></p>
<p align='left'><font size="6px">Corpus: I like deep learning. I like NLP. I enjoy flying.</font></p>
<p align='left'><font size="6px">Dictionary: [ 'I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.' ]</font></p>
</div>
<div style='width:50%;float:left;height:600px;'>
<img src="https://editor.analyticsvidhya.com/uploads/37840Screenshot%202021-06-18%20at%2012.19.48%20AM.png">
</div>

<!-- slide -->
<p align='left'><font size="20px">Self-supervised 
word2vec</font></p>
<p align='left'><font size="6px"><a href=https://code.google.com/archive/p/word2vec/ target="_blank">word2vec</a>工具是为了解决上述问题而提出的。它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。word2vec工具包含两个模型，即跳元模型（<a href=https://arxiv.org/abs/1310.4546 target="_blank">skip-gram</a>）和连续词袋（<a href=https://arxiv.org/abs/1301.3781 target="_blank">CBOW</a>）。其后有各种优化版本的词向量出现，比较广泛应用的有<a href=https://nlp.stanford.edu/pubs/glove.pdf target="_blank">glove</a>和<a href=https://arxiv.org/abs/1607.04606 target="_blank">fasttext</a>。</font></p>
<div style='width:50%;float:left;height:300px;'>The Skip-Gram Model
<font size="6px">$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$</font>
<img src="https://zh-v2.d2l.ai/_images/skip-gram.svg" width=50% style="background-color:white;">
<p><font size="5px">$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}, v_i \in \mathbb{R}^d \& u_i \in \mathbb{R}^d$</font>
<p><font size="5px">$\arg\max\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$</font>
</div>
<div style='width:50%;float:left;height:300px;'>The CBOW Model
<font size="6px">$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$</font>
<img src="https://zh-v2.d2l.ai/_images/cbow.svg" width=50% style="background-color:white;">
<p><font size="5px">$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)},v_i \in \mathbb{R}^d \& u_i \in \mathbb{R}^d$</font>
<p><font size="5px">$\arg\max\prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$</font>
</div>

<!-- slide -->
<p align='left'><font size="20px">Self-supervised 
word2vec</font></p>
<div style='width:49%;float:left;height:600px;'>The Skip-Gram Model
<p><font size="5px">$\text{loss} = - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)})$</font>
<p><font size="5px">$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right)$</font>
<p><font size="5px">$\begin{split}\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j\end{aligned}\end{split}$</font>
<p align='left'><font size="5px">对词典中索引为$i$的词进行训练后，得到$v_i$（作为中心词）和$u_i$（作为上下文词）两个词向量。在自然语言处理应用中，跳元模型的中心词向量通常用作词表示。</font>
</div>
<div style='width:2px;border:1px solid white;float:left;height:600px;'></div>
<div style='width:49%;float:left;height:600px;'>The CBOW Model
<p><font size="5px">$\text{loss} = -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$</font>
<p><font size="5px">$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right)$</font>S
<p><font size="5px">$\begin{split}\begin{aligned}\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} &= \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right)\\&= \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right)\end{aligned}\end{split}$</font>
<p align='left'><font size="5px">其他词向量的梯度可以以相同的方式获得。与跳元模型不同，连续词袋模型通常使用上下文词向量作为词表示。</font>
</div>

<!-- slide -->
<p align='left'><font size="20px">Approximate Training</font></p>
<div style='width:100%;float:left;height:600px;'>
<p align='left'><font size="6px">Negative sampling
<p align='left'><font size="5px">负采样修改了原目标函数。给定中心词$w_c$的上下文窗口，任意上下文词$w_o$来自该上下文窗口的被认为是由下式建模概率的事件:</font></p>
<font size="5px">$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c)$$</font>
<p align='left'><font size="5px">用$S$表示上下文词$w_o$来自中心词$w_c$的上下文窗口的事件。对于这个涉及$w_o$的事件，从预定义分布$P(w)$中采样$K$个不是来自这个上下文窗口噪声词。用$N_k$表示噪声词$w_k\,(k=1,…,K)$不是来自$w_c$的上下文窗口的事件。</font></p>
<font size="5px">$$\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$$</font>
<font size="5px">$$P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k)$$</font>
</div>

<!-- slide -->
<p align='left'><font size="20px">Approximate Training</font></p>
<div style='width:100%;float:left;height:600px;'>
<p align='left'><font size="6px">Negative sampling

<font size="5px">$$\begin{split}\begin{aligned}
l=-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}\end{split}$$
$$\frac{\partial l}{v_{i_t}} = -\sigma\left(-\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) \mathbf{u}_{i_{t+j}} + \sum_{k=1, w_k \sim P(w)}^K \sigma \left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t} \right) \mathbf{u}_{h_k}$$
</font></div>

<!-- slide -->
<img src=https://blog.acolyer.org/wp-content/uploads/2016/04/word2vec-dr-fig-2.png height=70% width=70%></img>
<!-- slide -->
<p align='left'><a href=https://arxiv.org/abs/1802.05365 target="_blank"><font size="20px">ELMo</font></a></p>
<div style='width:50%;float:left;height:200px;'>
<p align='left'><font size="6px">word2vec和GloVe都将相同的预训练向量分配给同一个词，而不考虑词的上下文（如果有的话）。单身人的来由：原来是喜欢一个人；现在是喜欢一个人。“一个人”一词有完全不同的含义；因此，同一个词可以根据上下文被赋予不同的表示。</font></p>
</div>
<div style='width:50%;float:left;height:200px;'>
<img src=https://upload.wikimedia.org/wikipedia/commons/5/5a/Jeff_Sessions_with_Elmo_and_Rosita_%28cropped%29.jpg></img>
</div>
<div style='width:100%;float:left;height:20px;'></div>
<div style='width:100%;float:left;height:600px;'>
<img src=https://pic2.zhimg.com/v2-7cc35961aa9a92d9134fc80ad890dccc_1440w.jpg?source=172ae18b height=70% width=50%></img>
<font size="6px">$\sum_{k=1}^N (log p(t_{k}|t_{1},...,t_{k-1};\Theta_{x},\overrightarrow\Theta_{LSTM},\Theta_{s})+log p(t_{k}|t_{k+1},...,t_{N};\overleftarrow\Theta_{LSTM},\Theta_{s}))$</font>
</div>

<!-- slide -->
<p align='left'><a href=https://arxiv.org/abs/1802.05365 target="_blank"><font size="20px">ELMo</font></a></p>
<div style='width:100%;float:left;height:20px;'></div>
<div style='width:100%;float:left;height:600px;'>
<img src=https://paddlepedia.readthedocs.io/en/latest/_images/elmo.png height=100% width=80%></img>
</div>
