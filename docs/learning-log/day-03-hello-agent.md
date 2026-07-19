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

## Simple Self-Attention Formulas
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

## 6. PyTorch equivalent

```python
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
all_context_vecs = attn_weights @ inputs
```

In Section 3.3.2, input embeddings play all three roles: query, key, and value. Section 3.4 introduces learned Query, Key, and Value projections.








### Strength and weakness of attention