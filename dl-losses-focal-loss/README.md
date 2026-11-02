# Focal Loss

Intermediate | deep-learning | losses | classification

### The problem, from first principles

Focal loss addresses class imbalance by down-weighting easy examples and focusing on hard ones. It is particularly useful for object detection where background is dominant.

$$\text{focal}(p_t, \gamma=2.0) = -(1-p_t)^\gamma \log(p_t)$$

Where p_t is the model's probability of the ground truth class. When p_t is high (easy), the weight (1-p_t)^gamma is small. When p_t is low (hard), the weight is large.

### From theory to code

Implement:

```python
focal_loss(logits, labels, gamma=2.0)
```

Where logits are raw model outputs, labels are 0/1 class labels.

### Constraints

- logits and labels same shape.
- gamma >= 0.
- Return scalar loss.
