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

## 6. PyTorch equivalent

```python
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
all_context_vecs = attn_weights @ inputs
```

In Section 3.3.2, input embeddings play all three roles: query, key, and value. Section 3.4 introduces learned Query, Key, and Value projections.
### Strength and weakness of attention