# Day 3 — LLM Bridge

**Date:** 2026-07-18
**Study time:** 2 hours


## Attention
self-attention mechanism
### Definition
self attention is a key conomponent of contemporary LLMs base on Transformer architecture

### Why Attention
Self attention will consider the relevency when handle the input text, and find the most relevent text.

### attention machanism required element
1. context vector

### Implementation steps of self-attention
1. Compute the attention socore
2. Normalize the attention score (Softmax function or others)
3. Attention weight should sum up to 1
4. calculating the context vector

## Simple Self-Attention Formulas(without trianable weight)
Sentence: `Your journey starts with one step`

We calculate the context vector for the second token, `journey`.

## Input embeddings

$$
x^{(1)}, x^{(2)}, \ldots, x^{(T)}, \qquad x^{(i)} \in \mathbb{R}^{d}
$$

For this example, $T=6$ tokens and $d=3$ embedding dimensions.

$$
x^{(1)} =
\begin{bmatrix} 0.43 & 0.15 & 0.89 \end{bmatrix}
$$

$$
x^{(2)} =
\begin{bmatrix} 0.55 & 0.87 & 0.66 \end{bmatrix}
$$

$$
x^{(3)} =
\begin{bmatrix} 0.57 & 0.85 & 0.64 \end{bmatrix}
$$

$$
x^{(4)} =
\begin{bmatrix} 0.22 & 0.58 & 0.33 \end{bmatrix}
$$

$$
x^{(5)} =
\begin{bmatrix} 0.77 & 0.25 & 0.10 \end{bmatrix}
$$

$$
x^{(6)} =
\begin{bmatrix} 0.05 & 0.80 & 0.55 \end{bmatrix}
$$

The query is the second token:

$$
q^{(2)} = x^{(2)} =
\begin{bmatrix} 0.55 & 0.87 & 0.66 \end{bmatrix}
$$

## Attention score

$$
\omega_{ij} = x^{(i)} \left(x^{(j)}\right)^\top
$$

For `journey` against `Your`:

$$
\omega_{21} = x^{(1)}\left(x^{(2)}\right)^\top
$$

$$
= (0.43 \times 0.55) + (0.15 \times 0.87) + (0.89 \times 0.66)
$$

$$
= 0.2365 + 0.1305 + 0.5874 = 0.9544
$$

All scores for the second token are:

$$
\omega_{2,:} =
\begin{bmatrix}
0.9544 & 1.4950 & 1.4754 & 0.8434 & 0.7070 & 1.0865
\end{bmatrix}
$$

## Attention weight

$$
\alpha_{ij} =
\frac{\exp(\omega_{ij})}
{\displaystyle\sum_{k=1}^{T}\exp(\omega_{ik})}
$$

$$
\alpha_{2,:} =
\operatorname{softmax}
\left(
\begin{bmatrix}
0.9544 & 1.4950 & 1.4754 & 0.8434 & 0.7070 & 1.0865
\end{bmatrix}
\right)
$$

$$
=
\begin{bmatrix}
0.1385 & 0.2379 & 0.2333 & 0.1240 & 0.1082 & 0.1581
\end{bmatrix}
$$

## Normalization

$$
0 \leq \alpha_{ij} \leq 1
$$

$$
\sum_{j=1}^{T}\alpha_{ij} = 1
$$

$$
0.1385 + 0.2379 + 0.2333 + 0.1240 + 0.1082 + 0.1581 = 1.0
$$

## Context vector

$$
z^{(i)} = \sum_{j=1}^{T}\alpha_{ij}x^{(j)}
$$

$$
z^{(2)} =
\alpha_{21}x^{(1)} + \alpha_{22}x^{(2)} + \alpha_{23}x^{(3)}
 + \alpha_{24}x^{(4)} + \alpha_{25}x^{(5)} + \alpha_{26}x^{(6)}
$$

$$
z^{(2)} =
0.1385\begin{bmatrix} 0.43 & 0.15 & 0.89 \end{bmatrix}
+ 0.2379\begin{bmatrix} 0.55 & 0.87 & 0.66 \end{bmatrix}
+ 0.2333\begin{bmatrix} 0.57 & 0.85 & 0.64 \end{bmatrix}
$$

$$
+ 0.1240\begin{bmatrix} 0.22 & 0.58 & 0.33 \end{bmatrix}
+ 0.1082\begin{bmatrix} 0.77 & 0.25 & 0.10 \end{bmatrix}
+ 0.1581\begin{bmatrix} 0.05 & 0.80 & 0.55 \end{bmatrix}
$$

$$
z^{(2)} =
\begin{bmatrix} 0.4419 & 0.6515 & 0.5683 \end{bmatrix}
$$

## Matrix form

$$
X =
\begin{bmatrix}
0.43 & 0.15 & 0.89 \\
0.55 & 0.87 & 0.66 \\
0.57 & 0.85 & 0.64 \\
0.22 & 0.58 & 0.33 \\
0.77 & 0.25 & 0.10 \\
0.05 & 0.80 & 0.55
\end{bmatrix}
$$

$$
\Omega = XX^\top
$$

The second row of the score matrix is:

$$
\Omega_{2,:} =
\begin{bmatrix}
0.9544 & 1.4950 & 1.4754 & 0.8434 & 0.7070 & 1.0865
\end{bmatrix}
$$

## Final result

```text
Original journey embedding:      [0.55,   0.87,   0.66]
Context-aware journey vector z²: [0.4419, 0.6515, 0.5683]
```

# Section 3.3.2: Self-Attention for All Tokens

Sentence: `Your journey starts with one step`

Section 3.3.1 calculated one context vector, $z^{(2)}$, for `journey`.
Section 3.3.2 performs the same calculation for **every token at once**.

## 1. Input embedding matrix

There are $T=6$ tokens and each embedding has $d=3$ values.

$$
X =
\begin{bmatrix}
0.43 & 0.15 & 0.89 \\
0.55 & 0.87 & 0.66 \\
0.57 & 0.85 & 0.64 \\
0.22 & 0.58 & 0.33 \\
0.77 & 0.25 & 0.10 \\
0.05 & 0.80 & 0.55
\end{bmatrix}
$$

Shape:

$$
X \in \mathbb{R}^{6 \times 3}
$$

Each row is one token embedding.

```text
row 1 = Your
row 2 = journey
row 3 = starts
row 4 = with
row 5 = one
row 6 = step
```

## 2. Attention-score matrix

The abstract formula is:

$$
\Omega = XX^\top
$$

The transpose changes the shape of $X$ from $6 \times 3$ to $3 \times 6$:

$$
X^\top \in \mathbb{R}^{3 \times 6}
$$

Therefore:

$$
(6 \times 3)(3 \times 6) = 6 \times 6
$$

Numerically:

$$
\Omega =
\begin{bmatrix}
0.9995 & 0.9544 & 0.9422 & 0.4753 & 0.4576 & 0.6310 \\
0.9544 & 1.4950 & 1.4754 & 0.8434 & 0.7070 & 1.0865 \\
0.9422 & 1.4754 & 1.4570 & 0.8296 & 0.7154 & 1.0605 \\
0.4753 & 0.8434 & 0.8296 & 0.4937 & 0.3474 & 0.6565 \\
0.4576 & 0.7070 & 0.7154 & 0.3474 & 0.6654 & 0.2935 \\
0.6310 & 1.0865 & 1.0605 & 0.6565 & 0.2935 & 0.9450
\end{bmatrix}
$$

Interpretation:

$$
\omega_{ij} = x^{(i)}\left(x^{(j)}\right)^\top
$$

- Row $i$ is the token acting as the query.
- Column $j$ is the token being compared with that query.
- $\omega_{ij}$ is the raw relevance score.

For example, the second row is the result already calculated for `journey`:

$$
\Omega_{2,:} =
\begin{bmatrix}
0.9544 & 1.4950 & 1.4754 & 0.8434 & 0.7070 & 1.0865
\end{bmatrix}
$$

## 3. Attention-weight matrix

Apply softmax **to each row** of the score matrix:

$$
A = \operatorname{softmax}(\Omega, \operatorname{dim}=-1)
$$

Each row of $A$ is a probability distribution:

$$
\sum_{j=1}^{T}\alpha_{ij} = 1
$$

Numerically:

$$
A =
\begin{bmatrix}
0.2098 & 0.2006 & 0.1981 & 0.1242 & 0.1220 & 0.1452 \\
0.1385 & 0.2379 & 0.2333 & 0.1240 & 0.1082 & 0.1581 \\
0.1390 & 0.2369 & 0.2326 & 0.1242 & 0.1108 & 0.1565 \\
0.1435 & 0.2074 & 0.2046 & 0.1462 & 0.1263 & 0.1720 \\
0.1526 & 0.1958 & 0.1975 & 0.1367 & 0.1879 & 0.1295 \\
0.1385 & 0.2184 & 0.2128 & 0.1420 & 0.0988 & 0.1896
\end{bmatrix}
$$

The second row means that `journey` assigns:

```text
13.85% to Your
23.79% to journey
23.33% to starts
12.40% to with
10.82% to one
15.81% to step
```

## 4. All context vectors

The abstract formula is:

$$
Z = AX
$$

The shapes are:

$$
(6 \times 6)(6 \times 3) = 6 \times 3
$$

Numerically:

$$
Z =
\begin{bmatrix}
0.4421 & 0.5931 & 0.5790 \\
0.4419 & 0.6515 & 0.5683 \\
0.4431 & 0.6496 & 0.5671 \\
0.4304 & 0.6298 & 0.5510 \\
0.4671 & 0.5910 & 0.5266 \\
0.4177 & 0.6503 & 0.5645
\end{bmatrix}
$$

Each row is a context-aware embedding:

```text
row 1 = context vector for Your
row 2 = context vector for journey
row 3 = context vector for starts
row 4 = context vector for with
row 5 = context vector for one
row 6 = context vector for step
```

The second row matches Section 3.3.1:

$$
z^{(2)} =
\begin{bmatrix}
0.4419 & 0.6515 & 0.5683
\end{bmatrix}
$$

## 5. Complete pipeline

$$
X
\xrightarrow{XX^\top}
\Omega
\xrightarrow{\operatorname{softmax}\text{ per row}}
A
\xrightarrow{AX}
Z
$$

```text
Input embeddings
→ compare every token with every other token
→ normalize each token's scores into attention weights
→ use each row of weights to mix all input embeddings
→ produce one context-aware vector for every token
```

# Section Computing the attention weights step by step
## Self-Attention Formulas(with trianable weight)
1. Compute the Q K V vector weight. (Using random(seed)and nn.rand())
2. Calculate the query vector
3. Calculate the key vector
4. Calculate the value vector
5. Calculate keys and values for the other tokens
6. Convert scores to attention weights
7. Calculate the context vector

Terminology
- **$W_Q, W_K, W_V$**: trainable weight matrices.
- **$q, k, v$**: vectors produced from an input using those matrices.
- **$Q, K, V$**: collections of query, key, and value vectors for all tokens.

The figure focuses on the second input token, **“journey”**:

$$
x^{(2)} = [0.5, 0.8, 0.6]
$$

Assume input vectors are row vectors:

$$
x^{(2)} \in \mathbb{R}^{1\times 3}
$$

The trainable matrices have shape:

$$
W_Q, W_K, W_V \in \mathbb{R}^{3\times 2}
$$

Therefore, each query, key, and value vector has two components.

> The exact matrices are not visible in the figure. The matrices below are one numerical example chosen to reproduce the displayed vectors.

```text
W_Q = [ [0.16, 0.56],
        [0.256, 0.896],
        [0.192, 0.672] ]

W_K = [ [0.16, 0.44],
        [0.256, 0.704],
        [0.192, 0.528] ]

W_V = [ [0.12, 0.40],
        [0.192, 0.64],
        [0.144, 0.48] ]
```

These matrices are trainable: during training, backpropagation adjusts their values.

## Step 1: Calculate the query vector

The query is:

$$
q^{(2)} = x^{(2)}W_Q
$$

Substitute the numbers:

$$
q^{(2)} =
[0.5, 0.8, 0.6]
\begin{bmatrix}
0.16 & 0.56\\
0.256 & 0.896\\
0.192 & 0.672
\end{bmatrix}
$$

### First component

Multiply the input by the first column of $W_Q$:

$$
q^{(2)}_1 =
(0.5)(0.16)+(0.8)(0.256)+(0.6)(0.192)
$$

$$
=0.08+0.2048+0.1152=0.4
$$

### Second component

Multiply the input by the second column of $W_Q$:

$$
q^{(2)}_2 =
(0.5)(0.56)+(0.8)(0.896)+(0.6)(0.672)
$$

$$
=0.28+0.7168+0.4032=1.4
$$

Therefore:

$$
\boxed{q^{(2)}=[0.4, 1.4]}
$$

The query represents what “journey” is looking for in the other tokens.

## Step 2: Calculate the key vector

$$
k^{(2)} = x^{(2)}W_K
$$

$$
k^{(2)} =
[0.5, 0.8, 0.6]
\begin{bmatrix}
0.16 & 0.44\\
0.256 & 0.704\\
0.192 & 0.528
\end{bmatrix}
$$

### First component

$$
k^{(2)}_1 =
(0.5)(0.16)+(0.8)(0.256)+(0.6)(0.192)
$$

$$
=0.08+0.2048+0.1152=0.4
$$

### Second component

$$
k^{(2)}_2 =
(0.5)(0.44)+(0.8)(0.704)+(0.6)(0.528)
$$

$$
=0.22+0.5632+0.3168=1.1
$$

Therefore:

$$
\boxed{k^{(2)}=[0.4, 1.1]}
$$

The key represents what kind of information “journey” contains.

## Step 3: Calculate the value vector

$$
v^{(2)} = x^{(2)}W_V
$$

$$
v^{(2)} =
[0.5, 0.8, 0.6]
\begin{bmatrix}
0.12 & 0.40\\
0.192 & 0.64\\
0.144 & 0.48
\end{bmatrix}
$$

### First component

$$
v^{(2)}_1 =
(0.5)(0.12)+(0.8)(0.192)+(0.6)(0.144)
$$

$$
=0.06+0.1536+0.0864=0.3
$$

### Second component

$$
v^{(2)}_2 =
(0.5)(0.40)+(0.8)(0.64)+(0.6)(0.48)
$$

$$
=0.20+0.512+0.288=1.0
$$

Therefore:

$$
\boxed{v^{(2)}=[0.3, 1.0]}
$$

The value contains information that may be passed to another token.

## Step 4: Calculate keys and values for other tokens

The same operations are performed for every token:

$$
k^{(i)}=x^{(i)}W_K,
\qquad
v^{(i)}=x^{(i)}W_V
$$

The figure gives approximately:

$$
k^{(1)}=[0.3, 0.7],
\qquad
v^{(1)}=[0.1, 0.8]
$$

for **“Your,”** and:

$$
k^{(T)}=[0.3, 0.9],
\qquad
v^{(T)}=[0.3, 0.7]
$$

for **“step.”** The intermediate tokens, represented by the dots, also have their own keys and values.

## Step 5: Compare the query with each key

The query for “journey” is compared with every key using a dot product:

$$
\omega_{2j}=q^{(2)}\cdot k^{(j)}
$$

### Score for “Your”

$$
\omega_{21}=[0.4, 1.4]\cdot[0.3, 0.7]
$$

$$
=(0.4)(0.3)+(1.4)(0.7)=0.12+0.98=1.10
$$

The figure shows approximately $1.2$ because its displayed vectors are rounded.

### Score for “journey”

$$
\omega_{22}=[0.4, 1.4]\cdot[0.4, 1.1]
$$

$$
=(0.4)(0.4)+(1.4)(1.1)=0.16+1.54=1.70
$$

The figure shows approximately $1.8$.

### Score for “step”

$$
\omega_{2T}=[0.4, 1.4]\cdot[0.3, 0.9]
$$

$$
=(0.4)(0.3)+(1.4)(0.9)=0.12+1.26=1.38
$$

The figure shows approximately $1.5$.

Using the rounded scores from the figure:

$$
[\omega_{21},\omega_{22},\omega_{2T}]=[1.2, 1.8, 1.5]
$$

A larger score means that “journey” considers the corresponding token more relevant.

## Step 6: Convert scores to attention weights

Softmax converts raw scores into numbers between 0 and 1 that add up to 1:

$$
\alpha_{2j}=
\frac{e^{\omega_{2j}}}{\sum_m e^{\omega_{2m}}}
$$

First calculate the exponentials:

$$
e^{1.2}\approx3.320,
\qquad
e^{1.8}\approx6.050,
\qquad
e^{1.5}\approx4.482
$$

Calculate the denominator:

$$
3.320+6.050+4.482=13.852
$$

Now calculate each weight:

$$
\alpha_{21}=\frac{3.320}{13.852}\approx0.240
$$

$$
\alpha_{22}=\frac{6.050}{13.852}\approx0.437
$$

$$
\alpha_{2T}=\frac{4.482}{13.852}\approx0.324
$$

Therefore:

$$
\boxed{[0.240, 0.437, 0.324]}
$$

These weights add up to approximately 1:

$$
0.240+0.437+0.324\approx1.001
$$

The small difference is caused by rounding.

## Step 7: Calculate the context vector

Use the attention weights to combine the value vectors:

$$
z^{(2)}=
\alpha_{21}v^{(1)}+
\alpha_{22}v^{(2)}+
\alpha_{2T}v^{(T)}
$$

Substitute the numbers:

$$
z^{(2)}=
0.240[0.1,0.8]
+0.437[0.3,1.0]
+0.324[0.3,0.7]
$$

### First component

$$
z^{(2)}_1=
(0.240)(0.1)+(0.437)(0.3)+(0.324)(0.3)
$$

$$
=0.024+0.1311+0.0972\approx0.252
$$

### Second component

$$
z^{(2)}_2=
(0.240)(0.8)+(0.437)(1.0)+(0.324)(0.7)
$$

$$
=0.192+0.437+0.2268\approx0.856
$$

Therefore:

$$
\boxed{z^{(2)}=[0.252, 0.856]}
$$

This is the context-aware representation of “journey.”

## What happens for all tokens?

The figure focuses on the second token, but the full model performs this calculation for every token:

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V
$$

All attention scores are then calculated at once:

$$
S=QK^\top
$$

Each row of $S$ corresponds to one query token:

- Row 1: how “Your” attends to all tokens.
- Row 2: how “journey” attends to all tokens.
- Row 3: how the third token attends to all tokens.
- And so on.

Finally:

$$
Z=\operatorname{softmax}(S)V
$$

Every token receives its own context vector.

The complete flow is:

$$
\boxed{
x
\overset{W_Q,W_K,W_V}{\longrightarrow}
q,k,v
\overset{qk^\top}{\longrightarrow}
\text{scores}
\overset{\text{softmax}}{\longrightarrow}
\text{attention weights}
\overset{\text{weighted sum of }v}{\longrightarrow}
\text{context vector}
}
$$

## causal attention

### What is causal attention
In causal attention, we mask out the attention weights above the diagonal such that for a given input, the LLM can’t access future tokens when computing the context vectors using the attention weights.
 
### Implementation of causal attention
1. compute the attention weights using the softmax function
2. create a maskwhere the values above the diagonal are zero(can multiple -inf)
3. multiply this mask with the attention weights
4. 


## 6. PyTorch equivalent

```python
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
all_context_vecs = attn_weights @ inputs
```
# Multi-Head Attention: A Numeric Two-Head Example

This document explains multi-head attention step by step using small numbers.
The example uses **two tokens**, **two attention heads**, and a model dimension of **four**.

The goal is to make every reshape, matrix multiplication, softmax calculation, and concatenation visible.

## 1. Configuration

| Symbol | Meaning | Value |
|---|---|---:|
| `B` | Batch size | `1` |
| `S` | Sequence length | `2` |
| `D` | Model dimension | `4` |
| `H` | Number of attention heads | `2` |
| `d_head` | Dimension per head | `D / H = 2` |

The model dimension is split across the heads:

```text
d_head = D / H = 4 / 2 = 2
```

Each token starts with a vector of length `4`. Each attention head works with a vector of length `2`.

## 2. Combined QKV projection

In a real Transformer, a learned linear layer creates queries, keys, and values:

```python
qkv = linear(x)
```

If `D = 4`, the linear layer usually outputs `3 * D = 12` values per token:

```text
[ Q (4 values) | K (4 values) | V (4 values) ]
```

For this worked example, assume the combined projection has produced:

```text
Token 1: [ 1, 0, 0, 1 |  1, 0, 0, 1 | 10, 20, 1, 2 ]
Token 2: [ 0, 1, 1, 0 |  0, 1, 1, 0 | 30, 40, 3, 4 ]
```

The shape is:

```text
(batch_size, sequence_length, 3 * model_dimension)
= (1, 2, 12)
```

## 3. Split the combined projection into Q, K, and V

The 12 values are split into three groups of 4:

```text
Q =
Token 1: [1, 0, 0, 1]
Token 2: [0, 1, 1, 0]

K =
Token 1: [1, 0, 0, 1]
Token 2: [0, 1, 1, 0]

V =
Token 1: [10, 20, 1, 2]
Token 2: [30, 40, 3, 4]
```

In matrix form:

```text
Q = ┌ 1  0  0  1 ┐
    └ 0  1  1  0 ┘

K = ┌ 1  0  0  1 ┐
    └ 0  1  1  0 ┘

V = ┌ 10  20  1  2 ┐
    └ 30  40  3  4 ┘
```

Each matrix has shape:

```text
(sequence_length, model_dimension) = (2, 4)
```

In PyTorch, the split is:

```python
Q, K, V = qkv.chunk(3, dim=-1)
```

## 4. Split Q, K, and V into two heads

Each vector has four values. Since there are two heads, each head receives two values.

### Head 1: first two dimensions

```text
Q1 = ┌ 1  0 ┐
     └ 0  1 ┘

K1 = ┌ 1  0 ┐
     └ 0  1 ┘

V1 = ┌ 10  20 ┐
     └ 30  40 ┘
```

### Head 2: last two dimensions

```text
Q2 = ┌ 0  1 ┐
     └ 1  0 ┘

K2 = ┌ 0  1 ┐
     └ 1  0 ┘

V2 = ┌ 1  2 ┐
     └ 3  4 ┘
```

The conceptual shape change is:

```text
(B, S, D) = (1, 2, 4)
        ↓ reshape
(B, S, H, d_head) = (1, 2, 2, 2)
        ↓ transpose
(B, H, S, d_head) = (1, 2, 2, 2)
```

The transpose places the head dimension before the sequence dimension so that attention can be calculated independently for each head.

## 5. Attention formula

For each head, scaled dot-product attention is:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_head) V
```

Here:

```text
d_head = 2
√d_head = √2 ≈ 1.4142
```

---

## 6. Calculate Head 1

### 6.1 Compute `Q1 K1ᵀ`

```text
Q1 = ┌ 1  0 ┐       K1ᵀ = ┌ 1  0 ┐
     └ 0  1 ┘              └ 0  1 ┘
```

Therefore:

```text
Q1 K1ᵀ = ┌ 1  0 ┐
          └ 0  1 ┘
```

### 6.2 Scale by `√d_head`

```text
(Q1 K1ᵀ) / √2

= ┌ 1 / 1.4142    0 / 1.4142 ┐
  └ 0 / 1.4142    1 / 1.4142 ┘

≈ ┌ 0.7071  0      ┐
  └ 0       0.7071 ┘
```

### 6.3 Apply softmax

For token 1:

```text
softmax([0.7071, 0]) ≈ [0.6698, 0.3302]
```

For token 2:

```text
softmax([0, 0.7071]) ≈ [0.3302, 0.6698]
```

Thus, the attention-weight matrix is:

```text
A1 = ┌ 0.6698  0.3302 ┐
     └ 0.3302  0.6698 ┘
```

Each row sums to approximately `1`.

### 6.4 Multiply the attention weights by `V1`

```text
V1 = ┌ 10  20 ┐
     └ 30  40 ┘
```

For token 1:

```text
[0.6698, 0.3302] V1

= 0.6698 [10, 20] + 0.3302 [30, 40]

≈ [16.604, 26.604]
```

For token 2:

```text
[0.3302, 0.6698] V1

= 0.3302 [10, 20] + 0.6698 [30, 40]

≈ [23.396, 33.396]
```

The output of Head 1 is:

```text
HeadOutput1 = ┌ 16.604  26.604 ┐
              └ 23.396  33.396 ┘
```

---

## 7. Calculate Head 2

### 7.1 Compute `Q2 K2ᵀ`

```text
Q2 = ┌ 0  1 ┐       K2ᵀ = ┌ 0  1 ┐
     └ 1  0 ┘              └ 1  0 ┘
```

Multiplying gives:

```text
Q2 K2ᵀ = ┌ 1  0 ┐
          └ 0  1 ┘
```

After scaling:

```text
(Q2 K2ᵀ) / √2

≈ ┌ 0.7071  0      ┐
  └ 0       0.7071 ┘
```

### 7.2 Apply softmax

```text
A2 = ┌ 0.6698  0.3302 ┐
     └ 0.3302  0.6698 ┘
```

### 7.3 Multiply by `V2`

```text
V2 = ┌ 1  2 ┐
     └ 3  4 ┘
```

For token 1:

```text
[0.6698, 0.3302] V2

= 0.6698 [1, 2] + 0.3302 [3, 4]

≈ [1.6604, 2.6604]
```

For token 2:

```text
[0.3302, 0.6698] V2

= 0.3302 [1, 2] + 0.6698 [3, 4]

≈ [2.3396, 3.3396]
```

The output of Head 2 is:

```text
HeadOutput2 = ┌ 1.6604  2.6604 ┐
              └ 2.3396  3.3396 ┘
```

## 8. Concatenate the heads

The two head outputs are joined along their last dimension.

For token 1:

```text
Head 1: [16.604, 26.604]
Head 2: [ 1.6604, 2.6604]

Combined: [16.604, 26.604, 1.6604, 2.6604]
```

For token 2:

```text
Head 1: [23.396, 33.396]
Head 2: [ 2.3396, 3.3396]

Combined: [23.396, 33.396, 2.3396, 3.3396]
```

The combined output is:

```text
Concat = ┌ 16.604  26.604  1.6604  2.6604 ┐
         └ 23.396  33.396  2.3396  3.3396 ┘
```

Its shape is again:

```text
(B, S, D) = (1, 2, 4)
```

## 9. Output projection

Normally, the concatenated output is passed through another learned linear layer:

```text
output = Concat @ W_O
```

where:

```text
W_O has shape (D, D) = (4, 4)
```

For simplicity, if we use the identity matrix:

```text
W_O = ┌ 1  0  0  0 ┐
      │ 0  1  0  0 │
      │ 0  0  1  0 │
      └ 0  0  0  1 ┘
```

then the final output is unchanged:

```text
Output = ┌ 16.604  26.604  1.6604  2.6604 ┐
         └ 23.396  33.396  2.3396  3.3396 ┘
```

In a real model, `W_O` mixes information from all heads.

## 10. Complete PyTorch implementation

```python
import math
import torch
import torch.nn.functional as F


# Shape: (batch_size, sequence_length, 3 * d_model)
#
# Each token is stored as:
# [Q (4 values) | K (4 values) | V (4 values)]
qkv = torch.tensor([
    [
        [1., 0., 0., 1.,   1., 0., 0., 1.,   10., 20., 1., 2.],
        [0., 1., 1., 0.,   0., 1., 1., 0.,   30., 40., 3., 4.],
    ]
])

batch_size = 1
seq_len = 2
d_model = 4
num_heads = 2
head_dim = d_model // num_heads


# Step 1: Split the combined projection into Q, K, and V.
#
# Before splitting: (B, S, 3D) = (1, 2, 12)
# After splitting:  each tensor is (B, S, D) = (1, 2, 4)
Q, K, V = qkv.chunk(3, dim=-1)


def split_heads(x):
    """Convert (B, S, D) into (B, H, S, d_head)."""
    B, S, D = x.shape

    # (B, S, D) -> (B, S, H, d_head)
    x = x.view(B, S, num_heads, head_dim)

    # (B, S, H, d_head) -> (B, H, S, d_head)
    return x.transpose(1, 2)


# Step 2: Split Q, K, and V into heads.
Q = split_heads(Q)
K = split_heads(K)
V = split_heads(V)

# Shapes are now:
# Q: (1, 2, 2, 2)
# K: (1, 2, 2, 2)
# V: (1, 2, 2, 2)


# Step 3: Calculate scaled dot-product attention scores.
# K.transpose(-2, -1) changes K from:
# (B, H, S, d_head) to (B, H, d_head, S)
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)


# Step 4: Convert scores into attention probabilities.
attention_weights = F.softmax(scores, dim=-1)


# Step 5: Use the attention probabilities to combine values.
# Result: (B, H, S, d_head)
head_outputs = attention_weights @ V


# Step 6: Move the head dimension back after the sequence dimension.
# (B, H, S, d_head) -> (B, S, H, d_head)
head_outputs = head_outputs.transpose(1, 2)


# Step 7: Combine all heads.
# (B, S, H, d_head) -> (B, S, D)
output = head_outputs.contiguous().view(
    batch_size,
    seq_len,
    d_model,
)

print("Attention weights:")
print(attention_weights)

print("Final output:")
print(output)
```

Expected final output, approximately:

```text
tensor([
    [
        [16.6040, 26.6040,  1.6604,  2.6604],
        [23.3960, 33.3960,  2.3396,  3.3396]
    ]
])
```

## 11. Shape summary

| Operation | Shape |
|---|---|
| Input `x` | `(1, 2, 4)` |
| Combined `qkv` | `(1, 2, 12)` |
| Each of `Q`, `K`, `V` | `(1, 2, 4)` |
| After splitting into heads | `(1, 2, 2, 2)` |
| After transposing heads | `(1, 2, 2, 2)` |
| Attention scores | `(1, 2, 2, 2)` |
| Head outputs | `(1, 2, 2, 2)` |
| After concatenating heads | `(1, 2, 4)` |
| Final output | `(1, 2, 4)` |

## 12. Main idea

The full process is:

```text
Input
  ↓
One linear projection creates Q, K, and V
  ↓
Split Q, K, and V into two heads
  ↓
Each head calculates its own attention
  ↓
Concatenate the head outputs
  ↓
Apply the output projection
  ↓
Final representation for every token
```

The most important lines in the implementation are:

```python
Q, K, V = qkv.chunk(3, dim=-1)
```

This separates queries, keys, and values.

```python
x = x.view(B, S, num_heads, head_dim)
```

This splits the model dimension across the heads.

```python
x = x.transpose(1, 2)
```

This puts the head dimension in the position needed for parallel attention.

```python
head_outputs = attention_weights @ V
```

This creates a weighted combination of value vectors.

```python
output = head_outputs.contiguous().view(B, S, d_model)
```

This joins all head outputs back into the original model dimension.



In Section 3.3.2, input embeddings play all three roles: query, key, and value. Section 3.4 introduces learned Query, Key, and Value projections.
### Strength and weakness of attention