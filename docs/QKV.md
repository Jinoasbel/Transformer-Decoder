# QKV in Attention Blocks

An attention block contains one or more attention heads. Each head has its own
Query (`W_Q`), Key (`W_K`), and Value (`W_V`) weight matrices.

For simplicity, this example uses only the **first input token**.

---

# Input Embeddings

$$
Input\ Vectors =
\begin{bmatrix}
1 & 2 & 3 \\
3 & 4 & 2 \\
7 & 2 & 1
\end{bmatrix}
$$

The first token embedding is

$$
Input_1=
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
$$

---

# Weight Matrices

## Query Weight

$$
Q_w=
\begin{bmatrix}
2 & 2 & 2 \\
1 & 6 & 8 \\
3 & 9 & 2
\end{bmatrix}
$$

## Key Weight

$$
K_w=
\begin{bmatrix}
6 & 2 & 8 \\
2 & 2 & 0 \\
2 & 1 & 9
\end{bmatrix}
$$

## Value Weight

$$
V_w=
\begin{bmatrix}
3 & 1 & 1 \\
4 & 2 & 9 \\
5 & 2 & 5
\end{bmatrix}
$$

---

# Step 1 : Calculate the Query Vector

The Query vector is obtained by multiplying the input embedding with the Query
weight matrix.

$$
Q_1 =
Input_1 \times Q_w
$$

Substituting the values,

$$
= \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\times
\begin{bmatrix}
2 & 2 & 2 \\
1 & 6 & 8 \\
3 & 9 & 2
\end{bmatrix}
$$

Matrix multiplication

---

### First Element

$$
(1\times2)+(2\times1)+(3\times3)
$$

$$
= 2+2+9
$$

$$
=13
$$

---

### Second Element

$$
(1\times2)+(2\times6)+(3\times9)
$$

$$
= 2+12+27
$$

$$
=41
$$

---

### Third Element

$$
(1\times2)+(2\times8)+(3\times2)
$$

$$
=2+16+6
$$

$$
=24
$$

---

Therefore,

$$
Q_1=
\begin{bmatrix}
13 & 41 & 24
\end{bmatrix}
$$

---

# Step 2 : Calculate the Key Vector

$$
K_1=
Input_1\times K_w
$$

$$
= \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\times
\begin{bmatrix}
6 & 2 & 8 \\
2 & 2 & 0 \\
2 & 1 & 9
\end{bmatrix}
$$

---

### First Element

$$
(1\times6)+(2\times2)+(3\times2)
$$

$$
= 6+4+6
$$

$$
=16
$$

---

### Second Element

$$
(1\times2)+(2\times2)+(3\times1)
$$

$$
= 2+4+3
$$

$$
=9
$$

---

### Third Element

$$
(1\times8)+(2\times0)+(3\times9)
$$

$$
= 8+0+27
$$

$$
=35
$$

---

Therefore,

$$
K_1= \begin{bmatrix}
16 & 9 & 35
\end{bmatrix}
$$

---

# Step 3 : Calculate the Value Vector

$$
V_1= Input_1\times V_w
$$

$$
= \begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
\times
\begin{bmatrix}
3 & 1 & 1 \\
4 & 2 & 9 \\
5 & 2 & 5
\end{bmatrix}
$$

---

### First Element

$$
(1\times3)+(2\times4)+(3\times5)
$$

$$
= 3+8+15
$$

$$
=26
$$

---

### Second Element

$$
(1\times1)+(2\times2)+(3\times2)
$$

$$
= 1+4+6
$$

$$
=11
$$

---

### Third Element

$$
(1\times1)+(2\times9)+(3\times5)
$$

$$
= 1+18+15
$$

$$
=34
$$

---

Therefore,

$$
V_1= \begin{bmatrix}
26 & 11 & 34
\end{bmatrix}
$$

---

# Step 4 : Calculate the Attention Score

The attention score is obtained by multiplying the Query vector with the
transpose of the Key vector.

$$
Score = Q_1K_1^T
$$

Substituting the values,

$$
= \begin{bmatrix}
13 & 41 & 24
\end{bmatrix}
\times
\begin{bmatrix}
16\\
9\\
35
\end{bmatrix}
$$

Since the first matrix is of size

$$
1\times3
$$

and the second matrix is

$$
3\times1,
$$

the output will be a

$$
1\times1
$$

matrix.

Now multiply each corresponding element.

$$
=(13\times16)+(41\times9)+(24\times35)
$$

Compute each product separately.

$$
13\times16=208
$$

$$
41\times9=369
$$

$$
24\times35=840
$$

Now add them together.

$$
208+369+840
$$

$$
=577+840
$$

$$
=1417
$$

Therefore,

$$
Score=
\begin{bmatrix}
1417
\end{bmatrix}
$$

---

# Step 5 : Apply Softmax

Since there is only **one score**, the softmax is

$$
Softmax(1417)=1
$$

Therefore,

$$
Attention\ Weight=
\begin{bmatrix}
1
\end{bmatrix}
$$

---

# Step 6 : Compute the Attention Output

The attention output is the weighted sum of the Value vectors.

Since the attention weight is

$$
1,
$$

the output becomes

$$
Output=
1\times
\begin{bmatrix}
26 & 11 & 34
\end{bmatrix}
$$

Therefore,

$$
Output=
\begin{bmatrix}
26 & 11 & 34
\end{bmatrix}
$$

---

# Final Result

### Query

$$
Q_1=
\begin{bmatrix}
13 & 41 & 24
\end{bmatrix}
$$

### Key

$$
K_1=
\begin{bmatrix}
16 & 9 & 35
\end{bmatrix}
$$

### Value

$$
V_1=
\begin{bmatrix}
26 & 11 & 34
\end{bmatrix}
$$

### Attention Score

$$
\begin{bmatrix}
1417
\end{bmatrix}
$$

### Attention Weight

$$
\begin{bmatrix}
1
\end{bmatrix}
$$

### Attention Output

$$
\begin{bmatrix}
26 & 11 & 34
\end{bmatrix}
$$

---

# Processing the Second Token

The second input embedding is

$$
Input_2=
\begin{bmatrix}
3 & 4 & 2
\end{bmatrix}
$$

---

## Step 1 : Calculate the Query Vector

$$
Q_2= Input_2\times Q_w
$$

$$
= \begin{bmatrix}
3 & 4 & 2
\end{bmatrix}
\times
\begin{bmatrix}
2 & 2 & 2 \\
1 & 6 & 8 \\
3 & 9 & 2
\end{bmatrix}
$$

### First Element

$$
(3\times2)+(4\times1)+(2\times3)
$$

$$
= 6+4+6
$$

$$
=16
$$

### Second Element

$$
(3\times2)+(4\times6)+(2\times9)
$$

$$
= 6+24+18
$$

$$
=48
$$

### Third Element

$$
(3\times2)+(4\times8)+(2\times2)
$$

$$
= 6+32+4
$$

$$
=42
$$

Therefore,

$$
Q_2= \begin{bmatrix}
16 & 48 & 42
\end{bmatrix}
$$

---

## Step 2 : Calculate the Key Vector

$$
K_2= Input_2\times K_w
$$

$$
= \begin{bmatrix}
3 & 4 & 2
\end{bmatrix}
\times
\begin{bmatrix}
6 & 2 & 8 \\
2 & 2 & 0 \\
2 & 1 & 9
\end{bmatrix}
$$

### First Element

$$
(3\times6)+(4\times2)+(2\times2)
$$

$$
= 18+8+4
$$

$$
=30
$$

### Second Element

$$
(3\times2)+(4\times2)+(2\times1)
$$

$$
= 6+8+2
$$

$$
=16
$$

### Third Element

$$
(3\times8)+(4\times0)+(2\times9)
$$

$$
= 24+0+18
$$

$$
=42
$$

Therefore,

$$
K_2= \begin{bmatrix}
30 & 16 & 42
\end{bmatrix}
$$

---

## Step 3 : Calculate the Value Vector

$$
V_2= Input_2\times V_w
$$

$$
= \begin{bmatrix}
3 & 4 & 2
\end{bmatrix}
\times
\begin{bmatrix}
3 & 1 & 1 \\
4 & 2 & 9 \\
5 & 2 & 5
\end{bmatrix}
$$

### First Element

$$
(3\times3)+(4\times4)+(2\times5)
$$

$$
= 9+16+10
$$

$$
=35
$$

### Second Element

$$
(3\times1)+(4\times2)+(2\times2)
$$

$$
= 3+8+4
$$

$$
=15
$$

### Third Element

$$
(3\times1)+(4\times9)+(2\times5)
$$

$$
= 3+36+10
$$

$$
=49
$$

Therefore,

$$
V_2= \begin{bmatrix}
35 & 15 & 49
\end{bmatrix}
$$

---

# Compute Attention Scores

The second token can attend to

- Token 1
- Token 2

Therefore,

$$
Scores=
Q_2K^T
$$

where

$$
Q_2=
\begin{bmatrix}
16 & 48 & 42
\end{bmatrix}
$$

and

$$
K=
\begin{bmatrix}
16&9&35\\
30&16&42
\end{bmatrix}
$$

Therefore,

$$
K^T=
\begin{bmatrix}
16&30\\
9&16\\
35&42
\end{bmatrix}
$$

Now multiply

$$
\begin{bmatrix}
16&48&42
\end{bmatrix}
\times
\begin{bmatrix}
16&30\\
9&16\\
35&42
\end{bmatrix}
$$

---

### Score with Token 1

$$
(16\times16)+(48\times9)+(42\times35)
$$

$$
= 256+432+1470
$$

$$
=2158
$$

---

### Score with Token 2

$$
(16\times30)+(48\times16)+(42\times42)
$$

$$
= 480+768+1764
$$

$$
=3012
$$

Therefore,

$$
Scores=
\begin{bmatrix}
2158 & 3012
\end{bmatrix}
$$

---

# Apply Softmax

Assume the softmax produces

$$
Softmax(Scores) =\begin{bmatrix}
0.30 & 0.70
\end{bmatrix}
$$

(The exact values are omitted because the numbers are extremely large.)

---

# Compute the Attention Output

Multiply each Value vector by its attention weight.

$$
0.30 \times \begin{bmatrix}
26&11&34
\end{bmatrix} =
\begin{bmatrix}
7.8&3.3&10.2
\end{bmatrix}
$$

$$
0.70\times \begin{bmatrix}
35&15&49
\end{bmatrix} =
\begin{bmatrix}
24.5&10.5&34.3
\end{bmatrix}
$$

Add the weighted Value vectors.

$$
\begin{bmatrix}
7.8&3.3&10.2
\end{bmatrix} +
\begin{bmatrix}
24.5&10.5&34.3
\end{bmatrix}
$$

$$
= \begin{bmatrix}
32.3&13.8&44.5
\end{bmatrix}
$$

Therefore,

$$
Output_2= \begin{bmatrix}
32.3&13.8&44.5
\end{bmatrix}
$$

---

# Processing the Third Token

Repeat the exact same process.

1. Compute

$$
Q_3=Input_3W_Q
$$

2. Compute

$$
K_3=Input_3W_K
$$

3. Compute

$$
V_3=Input_3W_V
$$

Append the new Key and Value to the previous ones.

$$
K=
\begin{bmatrix}
K_1\\
K_2\\
K_3
\end{bmatrix}
$$

$$
V=
\begin{bmatrix}
V_1\\
V_2\\
V_3
\end{bmatrix}
$$

Now compute

$$
Scores=
Q_3K^T
$$

which produces three attention scores.

$$
Scores=
\begin{bmatrix}
Q_3\cdot K_1 &
Q_3\cdot K_2 &
Q_3\cdot K_3
\end{bmatrix}
$$

Apply Softmax to obtain three attention weights.

$$
\begin{bmatrix}
w_1&w_2&w_3
\end{bmatrix}
$$

Finally compute the weighted sum of all Value vectors.

$$
Output_3=
w_1V_1+w_2V_2+w_3V_3
$$

This output is then passed to the next layer of the Transformer.
