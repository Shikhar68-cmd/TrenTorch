# stable-softmax-logits — Netflix case

Beginner | problemset | maths-stats-for-ml | probability-foundations | netflix

The team needs a reliable implementation for score-to-probability conversion in a production-oriented ML workflow. A small implementation error can change the model input or metric enough to make a downstream result misleading.

Given a vector of model logits, convert them into probabilities without overflowing when logits are large.

Use the maximum-subtraction form of softmax so that logits in the thousands do not overflow.

Implement `solve(logits)`.

**Returns.** Return a NumPy vector of probabilities with the same length as `logits`; the entries are positive and sum to 1.

### Examples

**Example 1**

Input:

```python
solve([1000.0, 1001.0])
```

Output:

```text
[0.268941, 0.731059]
```

**Example 2**

Input:

```python
solve([0.0, 0.0, 0.0])
```

Output:

```text
[0.333333, 0.333333, 0.333333]
```

**Example 3**

Input:

```python
solve([-1000.0, 0.0])
```

Output:

```text
[0.0, 1.0]
```
