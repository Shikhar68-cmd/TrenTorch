# attention-weighted-sum — Swiggy case

Beginner | problemset | sequence-models-attention | attention-mechanism | swiggy

Swiggy-inspired recommendation model combines several encoded signals using learned attention weights. You need to turn the raw attention scores into weights with a softmax and compute the weighted sum of the value vectors, so the model produces the intended context representation.

`scores` are the raw (unnormalised) attention scores for one query over $n$ items and `V` is the $n\times d$ matrix of their value vectors. Turn the scores into weights with a stable softmax and return the weighted sum of the rows of `V`.

Implement `solve(scores,V)`.

**Returns.** Return a float NumPy vector of length $d$.

### Examples

**Example 1**

Input:

```python
solve([0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]])
```

Output:

```text
[2.0, 3.0]
```

**Example 2**

Input:

```python
solve([10.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
```

Output:

```text
[0.999955, 4.5e-05]
```
