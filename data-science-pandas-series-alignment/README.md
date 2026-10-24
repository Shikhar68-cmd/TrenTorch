# Series, Index Alignment & Arithmetic

Beginner | data-science | pandas | series

### The problem, from first principles

A NumPy array knows its values and nothing else. A pandas `Series` carries a second thing with it, an **index** of labels, and that changes what arithmetic means. Add two arrays and the element at position 0 meets the element at position 0. Add two Series and the element labelled `"mon"` meets the element labelled `"mon"`, wherever each one happens to sit. This is called _index alignment_, and it is the reason a pandas calculation can silently give a different answer than the same numbers in an array: if a label appears on only one side, there is nothing to pair it with.

This question builds the four small operations that make alignment concrete: creating a labelled Series, adding two Series by label (with and without a fill value), turning values into shares of a total, and looking labels up without crashing on one that is not there.

### From theory to code

Implement `make_series(values, labels, name)`, `aligned_sum(a, b, fill_value=None)`, `share_of_total(s)` and `lookup(s, labels)`. The signatures and docstrings are already in the editor.

### Constraints

- `labels` are unique. `values` and `labels` have the same length.
- `make_series` returns a `pd.Series` with those values, that index (in the given order) and that name.
- `aligned_sum(a, b)` adds `a` and `b` by label. The result's index holds every label that appears in `a` or in `b`, sorted ascending. A label missing from one side gives `NaN`.
- `aligned_sum(a, b, fill_value=v)` treats a missing label **or** a `NaN` value on either side as `v` before adding. If both sides are missing or `NaN` for a label, the result for that label stays `NaN`.
- `aligned_sum` never changes `a` or `b`.
- `share_of_total(s)` returns a float Series with the same index and name as `s`, where each value is divided by the sum of the non-missing values. Missing values stay missing. The non-missing values do not sum to zero.
- `lookup(s, labels)` returns a Series indexed by `labels`, in that order, holding `s`'s value for each label and `NaN` for a label `s` does not have.

### Hints

<details>
<summary>Hint 1</summary>

Adding two Series with `+` already aligns on the index. What does it do to the order of the labels, and to a label present on one side only?

</details>

<details>
<summary>Hint 2</summary>

`Series.add` takes a `fill_value` argument that does the 'missing counts as v' step for you.

</details>

<details>
<summary>Hint 3</summary>

Selecting with `s[labels]` raises `KeyError` for an unknown label. A method that _conforms_ a Series to a new list of labels returns `NaN` instead.

</details>
