# Boolean Masks & Filtering

Beginner | data-science | pandas | filtering

### The problem, from first principles

Most of exploratory analysis is asking a question of a table and keeping the rows that answer it: customers over 30, orders from three cities, readings outside the normal range, rows where a value is _not_ missing. In pandas a question like that is a **boolean mask**, a column of `True`/`False` the same length as the table, and keeping the rows is just indexing with the mask.

The mask itself is built from comparisons, and combining comparisons has two traps that cost beginners hours. Python's `and`, `or` and `not` do not work on arrays (they ask for one truth value from many), so pandas uses `&`, `|` and `~`, and because of operator precedence every comparison must be wrapped in parentheses. The second trap is `NaN`: it is neither greater than nor less than anything, so a comparison against it is `False`, and `~(x > 5)` is `True` for a missing value, which is rarely what you meant.

### From theory to code

Implement `in_range(df, column, low, high)`, `from_cities_and_old_enough(df, cities, min_age)`, `any_extreme(df, column, low, high)`, `exclude_values(df, column, values)` and `complete_rows(df, columns)`. The signatures and docstrings are already in the editor.

### Constraints

- Every function returns the selected rows of `df` with their **original index labels and column order**, not renumbered.
- `in_range(df, column, low, high)` keeps rows where `low <= column <= high` (both ends included). Rows where the column is `NaN` are dropped.
- `from_cities_and_old_enough(df, cities, min_age)` keeps rows whose `city` is in `cities` **and** whose `age >= min_age`.
- `any_extreme(df, column, low, high)` keeps rows where `column < low` **or** `column > high`. A `NaN` is not extreme, so those rows are dropped.
- `exclude_values(df, column, values)` keeps rows whose `column` is **not** in `values`. A row where the column is `NaN` is kept (it is not in `values`).
- `complete_rows(df, columns)` keeps rows that have no missing value in any of the listed `columns`.
- Do not use `DataFrame.query`.

### Hints

<details>
<summary>Hint 1</summary>

`s.between(low, high)` is inclusive on both sides by default.

</details>

<details>
<summary>Hint 2</summary>

Wrap each comparison in parentheses before combining: `(df.a > 1) & (df.b < 2)`.

</details>

<details>
<summary>Hint 3</summary>

`s.isin(values)` gives a mask, and `~mask` flips it. `df[columns].notna().all(axis=1)` checks every listed column at once.

</details>
