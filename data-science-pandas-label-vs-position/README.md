# Label vs Position Selection

Beginner | data-science | pandas | indexing

### The problem, from first principles

A DataFrame can be addressed in two different languages. By **label**: the row called `"2024-03-01"`, the column called `"price"`. By **position**: the third row, the second column. The two agree only when the index happens to be `0, 1, 2, ...`, and the moment rows are filtered, sorted or re-indexed they stop agreeing. A bug that looks like "I selected the wrong rows" is nearly always a label used where a position was meant, or the reverse.

This question implements the selections you reach for every day, and each one is defined so you must know which language it speaks. They also settle two sharp edges: label slices include their end, position slices do not, and assigning into a _selection_ can silently change nothing or change the original frame, so the safe pattern is to work on a copy.

### From theory to code

Implement `rows_by_label(df, start, stop)`, `rows_by_position(df, start, stop)`, `cell(df, row_label, column)`, `numeric_columns(df)` and `with_value_where(df, mask, column, value)`. The signatures and docstrings are already in the editor.

### Constraints

- `df` has a unique, sorted-or-unsorted index of any type and unique column names.
- `rows_by_label(df, start, stop)` returns the rows from label `start` to label `stop` **inclusive of both ends**, in the frame's own order. Both labels exist in the index.
- `rows_by_position(df, start, stop)` returns rows at positions `start` up to but **not including** `stop`, like a Python slice.
- `cell(df, row_label, column)` returns the single value at that row label and column name.
- `numeric_columns(df)` returns the names of the columns holding numbers (integers or floats, not booleans), in the frame's column order, as a list.
- `with_value_where(df, mask, column, value)` returns a **new** frame in which `column` is set to `value` on the rows where the boolean `mask` is true. `df` itself is never changed.

### Hints

<details>
<summary>Hint 1</summary>

`.loc` is the label door and `.iloc` is the position door. Which one treats the end of a slice as included?

</details>

<details>
<summary>Hint 2</summary>

`select_dtypes` can pick columns by kind, and `'number'` is a kind it understands.

</details>

<details>
<summary>Hint 3</summary>

Copy first, then assign with `.loc[mask, column] = value`. Assigning to the original would change the caller's data.

</details>
