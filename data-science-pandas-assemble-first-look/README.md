# Assemble: A First Look at a Dataset

Intermediate | data-science | pandas | eda | assemble

### The problem, from first principles

The first thing anyone does with a new table is look at it, and the same handful of questions come up every time. How big is it? Which columns have holes, and how many? Are there repeated rows? For the numeric columns, what are the centre, the spread and the extremes? For the categorical ones, how many different values are there, and which one is most common? Notebooks answer with `df.info()` and `df.describe()` and a handful of `value_counts()` calls, and a person reads the output.

A program cannot read a printout, so this question builds the same first look as **data**: one dictionary a test (or a monitoring job, or a report) can check. It combines what the previous questions taught: dtypes, missing values, boolean logic, and sorting.

### From theory to code

Implement `profile(df)`. The signature and docstring are already in the editor.

### Constraints

- `profile(df)` returns a dict with exactly these keys: `n_rows`, `n_cols`, `missing`, `n_duplicate_rows`, `numeric`, `categorical`.
- `n_rows` and `n_cols` are Python `int`s. `missing` maps **every** column name to its number of missing values (an `int`), in column order.
- `n_duplicate_rows` is the number of rows that repeat an earlier row exactly (the first occurrence is not counted).
- A column is **numeric** if its dtype is integer or float (booleans are not numeric). Every other column is **categorical**.
- `numeric` maps each numeric column to a dict `{"mean", "std", "min", "max"}` of Python floats computed over the non-missing values. `std` is the **sample** standard deviation (divide by `n - 1`). If a column has fewer than 2 non-missing values, `std` is `nan`; if it has none, all four are `nan`.
- `categorical` maps each categorical column to `{"n_unique", "top", "top_count"}`: the number of distinct non-missing values, the most frequent non-missing value, and how many times it occurs. If several values tie for most frequent, `top` is the **smallest** of them. If the column has no non-missing values, `top` is `None` and `top_count` is `0`.

### Hints

<details>
<summary>Hint 1</summary>

`Series.isna().sum()` counts missing values, and `df.duplicated()` marks repeated rows.

</details>

<details>
<summary>Hint 2</summary>

`pd.api.types.is_numeric_dtype` is true for booleans too, so exclude them explicitly.

</details>

<details>
<summary>Hint 3</summary>

`value_counts()` sorts by count but its order among ties is not a contract you should rely on. Find the maximum count and take the smallest value that has it.

</details>
