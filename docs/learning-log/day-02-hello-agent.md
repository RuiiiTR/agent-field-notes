# Day 2 — LLM foundation

**Date:** 2026-07-14
**Study time:** 2 hours

# Chapter 3 

## ngrams
在 Bigram 模型里，"某个词在前一个词出现后出现的概率"由共现次数决定：

$$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1},\, w_i)}{\text{Count}(w_{i-1})}$$

一句话概括：词对一起出现的次数 ÷ 前一个词单独出现的总次数。

**为什么可以直接用次数？**

条件概率的原始定义是：

$$P(A \mid B) = \frac{P(A, B)}{P(B)}
= \frac{\text{Count}(A, B)\,/\,N}{\text{Count}(B)\,/\,N}
= \frac{\text{Count}(A, B)}{\text{Count}(B)}$$

其中 $N$ 为语料总规模。$N$ 在分子分母中约掉了，所以我们不用先算概率，直接数次数即可。这就是最大似然估计（MLE）。

```python
import collections

# 示例语料库，与上方案例讲解中的语料库保持一致
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# --- 第一步:计算 P(datawhale) ---
count_datawhale = tokens.count('datawhale')
p_datawhale = count_datawhale / total_tokens
print(f"第一步: P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

# --- 第二步:计算 P(agent|datawhale) ---
# 先计算 bigrams 用于后续步骤
bigrams = zip(tokens, tokens[1:])
bigram_counts = collections.Counter(bigrams)
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
# count_datawhale 已在第一步计算
p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f"第二步: P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}")

# --- 第三步:计算 P(learns|agent) ---
count_agent_learns = bigram_counts[('agent', 'learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f"第三步: P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}")

# --- 最后:将概率连乘 ---
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f"最后: P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}")

>>>
第一步: P(datawhale) = 2/6 = 0.333
第二步: P(agent|datawhale) = 2/2 = 1.000
第三步: P(learns|agent) = 1/2 = 0.500
最后: P('datawhale agent learns') ≈ 0.333 * 1.000 * 0.500 = 0.167
```

## cosin similarity
In n-grams, words are discrete symbols—two words are either equal or not, with no notion of "how similar" they are. Once converted into vectors, "similarity" acquires a computable definition.


$$
\text{similarity}(\vec{a}, \vec{b}) = \cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|\,|\vec{b}|}
$$

其中：

- $\vec{a} \cdot \vec{b}$ 是两个向量的**点积**（对应分量相乘再求和）。
- $|\vec{a}|$、$|\vec{b}|$ 是各自的**模长**，$|\vec{a}| = \sqrt{\sum_i a_i^2}$。
- 分母把长度归一化掉，所以结果只由**夹角** $\theta$ 决定，与向量大小无关。

取值范围 $[-1, 1]$：$1$ 方向完全相同，$0$ 正交，$-1$ 方向完全相反。

## 一个简单的计算

设 $\vec{a} = [3,\ 4]$，$\vec{b} = [4,\ 3]$。

**第 1 步：点积**

$$
\vec{a} \cdot \vec{b} = 3 \times 4 + 4 \times 3 = 12 + 12 = 24
$$

**第 2 步：各自的模长**

$$
|\vec{a}| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5, \qquad |\vec{b}| = \sqrt{4^2 + 3^2} = \sqrt{25} = 5
$$

**第 3 步：代入**

$$
\cos(\theta) = \frac{24}{5 \times 5} = \frac{24}{25} = 0.96
$$

结果 $0.96$ 接近 $1$，说明两向量方向很接近（夹角约 $16.26^\circ$）。

### tips
The conclusion still holds, but be clear: part of the credit for this elegant result belongs to hand-crafted filtering rules, not to the vector operations themselves. Using it to build the intuition that "relationships are computable" is fine; treating it as evidence that "word vectors can reliably perform logical reasoning" is an over-interpretation.

### temperature
# 模型采样参数：Temperature（温度）

在使用大模型时常见到 `Temperature` 这类可配置参数，其本质是**调整模型对概率分布的采样策略**，让输出匹配具体场景需求。

## 1. 出发点：Softmax 把 logit 变成概率

模型每一步对词表中每个候选 token 打一个原始分数 $z_i$（称 **logit**，可正可负、无概率含义）。Softmax 把它们压成一组和为 1 的概率：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
$$

指数保证全为正，分母做归一化。分数越高，概率越大。

## 2. 温度：在指数里加一个除数 $T$

温度采样把每个 logit 先除以 $T$（$T>0$）再进指数：

$$
p_i^{(T)} = \frac{e^{z_i/T}}{\sum_{j=1}^{k} e^{z_j/T}}
$$

- $T=1$：等于原始 Softmax，不改动分布。
- $T \to 0$：logit 差距被放到极大，最高分项碾压其它，分布极陡 → **确定、保守、易重复**。
- $T$ 变大：logit 被压缩，各项差距变小，分布变平坦，低概率词也有机会 → **多样、发散，但可能不连贯**。

### 具体感受（logit = [2, 1, 0]）

| $T$ | 三个 token 的概率 | 分布形态 |
|:---:|:---|:---|
| 0.5 | [0.867, 0.117, 0.016] | 更陡，第一名碾压 |
| 1.0 | [0.665, 0.245, 0.090] | 原始分布 |
| 2.0 | [0.506, 0.307, 0.186] | 更平，差距被抹平 |

## 3. 场景档位（经验值，非硬标准）

| 温度区间 | 输出风格 | 适用场景 |
|:---|:---|:---|
| 低温 `0 ≤ T < 0.3` | 精准、确定 | 问答、数据计算、代码生成、法律/技术文档、学术解释 |
| 中温 `0.3 ≤ T < 0.7` | 平衡、自然 | 客服、聊天机器人、邮件、文案、简单故事 |
| 高温 `0.7 ≤ T < 2` | 创新、发散 | 诗歌、科幻构思、广告 slogan、艺术灵感、发散思考 |

> 区间边界是经验惯例，不同模型/框架体感差异大，当"大致档位"用即可，别当精确阈值。

## 4. 五个容易被忽略、但很关键的点

1. **$T=0$ 不是代入公式**。公式里除以 0 未定义；工程上 `temperature=0` 被特判为 **argmax（贪心解码）**，是接口约定，不是真的把 0 代进分式。

2. **因果别讲反**。低温不是"直接把高概率项调大"，而是除以小 $T$ 把 logit 之间的**相对差距成倍拉大**，指数放大后高分项才碾压。若两个词 logit 本就接近，再低的温度也拉不开。

3. **温度不改变排序，只改变陡峭度**。除以正数 $T$ 不改变 logit 的大小顺序，所以概率排名永远不变，温度只调"第一名领先第二名多少"。这解释了为什么高温有时仍输出同一个词——第一名领先太多时压不平。

4. **温度不是唯一控制手段**。生产中常与 **top-k**（只在前 k 个词采样）、**top-p / 核采样**（只在累计概率达 p 的最小词集采样）配合或替代。图里说的"重新调整"对应温度，"截断"对应 top-k/top-p。一般不建议温度和 top-p 同时大幅调，否则难以归因——多数做法是固定 top_p、只动温度。

5. **低温 = 可复现 ≠ 正确**。低温让模型更坚定地选它最相信的答案；若模型本身错了，低温只会**稳定重复同一个错误**，不提高事实正确率。低温买的是一致性/可复现性，不直接等于准确。

## 5. 实用结论

温度本质是在调"**跟随模型的信心 vs. 探索模型的犹豫**"：

- 需要唯一确定答案（代码、抽取、分类、数学）→ 直接 `temperature=0`（贪心），别纠结 0.1 还是 0.2。
- 需要自然但可控（对话、改写）→ `0.5–0.7`，通常再配 `top_p ≈ 0.9`。
- 需要发散创意（头脑风暴、诗歌）→ `0.9–1.2`，接受偶尔跑偏，很少需要真开到 2。

## tokens
### Definition
LLM does not understand the sentence in a text. We need transform the sentence text into other format. In reality, sentence should be divided as some sort of number representations. 

### tokenization type
### BPE Tokenizer(Byte-Pair Encoding, BPE)
The implementation steps can be list as follows
1. The initial vaculary need to seprate to letter level
2. BPE then count the adjacen pairs frequency
3. If lo is the most frequent word, the lo combine as a token
4. repeat step 2,until we make sure all the letters are covered

### BPE Python implementation
```python
import re, collections

def get_stats(vocab):
    """统计词元对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """合并词元对"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 准备语料库，每个词末尾加上</w>表示结束，并切分好字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4 # 设置合并次数

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 20)

>>>
第1次合并: ('u', 'g') -> ug
新词表（部分）: ['h ug </w>', 'p ug </w>', 'p u n </w>', 'b u n </w>']
--------------------
第2次合并: ('ug', '</w>') -> ug</w>
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p u n </w>', 'b u n </w>']
--------------------
第3次合并: ('u', 'n') -> un
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un </w>', 'b un </w>']
--------------------
第4次合并: ('un', '</w>') -> un</w>
新词表（部分）: ['h ug</w>', 'p ug</w>', 'p un</w>', 'b un</w>']
--------------------
```
#### Strength and limitation of BPE
| Strength | Explanation |
|---|---|
| Balanced vocabulary size | Uses whole words, word pieces, and characters. |
| Handles rare words | Splits unknown words into smaller known pieces. |
| Fewer unknown tokens | Can represent new words instead of replacing them with `<UNK>`. |
| Efficient sequences | Common patterns become single tokens, reducing sequence length. |
| Supports many text types | Works with English, Chinese, code, JSON, symbols, and emojis. |
| Simple training | Learns tokens by repeatedly merging frequent adjacent pairs. |
| Reusable word pieces | Shared pieces such as `ing`, `tion`, or `agent` can appear in many words. |

| Limitation | Explanation |
|---|---|
| Not meaning-aware | Merges frequent pairs but does not understand semantics. |
| Unnatural splits | Words may be divided into unexpected pieces. |
| Unequal language efficiency | Different languages may require very different token counts. |
| Rare text is expensive | Names, formulas, code, and unusual symbols may use many tokens. |
| Model-specific | Different models use different vocabularies and merge rules. |
| Training-data dependent | Token quality depends on the corpus used to train the tokenizer. |
| Does not solve long context | Even with compression, long documents can exceed the context window. |
| Increases cost when token-heavy | More tokens mean more memory use, latency, and API cost. |
- tokens;
- context windows;
