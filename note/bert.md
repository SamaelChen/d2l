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
<p align='left'><font size="20px">Self-supervised 
word2vec</font></p>
<p align='left'><font size="6px"><a href=https://code.google.com/archive/p/word2vec/ target="_blank">word2vec</a>工具是为了解决上述问题而提出的。它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。word2vec工具包含两个模型，即跳元模型（<a href=https://arxiv.org/abs/1310.4546 target="_blank">skip-gram</a>）和连续词袋（<a href=https://arxiv.org/abs/1301.3781 target="_blank">CBOW</a>）。</font></p>
<div style='width:50%;float:left;height:600px;'>The Skip-Gram Model
<font size="6px">$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$</font>
<img src="https://zh-v2.d2l.ai/_images/skip-gram.svg" width=50% style="background-color:white;">
<p><font size="5px">$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}, v_i \in \mathbb{R}^d \& u_i \in \mathbb{R}^d$</font>
<p><font size="5px">$\arg\max\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$</font>
</div>
<div style='width:50%;float:left;height:600px;'>The CBOW Model
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
<div style='width:2px;border:1px solid white;float:left;height:700px;'></div>
<div style='width:49%;float:left;height:600px;'>The CBOW Model
<p><font size="5px">$\text{loss} = -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$</font>
<p><font size="5px">$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right)$</font>S
<p><font size="5px">$\begin{split}\begin{aligned}\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} &= \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right)\\&= \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right)\end{aligned}\end{split}$</font>
<p align='left'><font size="5px">其他词向量的梯度可以以相同的方式获得。与跳元模型不同，连续词袋模型通常使用上下文词向量作为词表示。</font>
</div>