# Sorting, Ranking & Top-N

Beginner | data-science | pandas | sorting | ranking

### The problem, from first principles

"Who are the top five?" is the commonest question asked of a table, and it hides three decisions. What happens when two rows tie for fifth place? Which rows go last when a value is missing? And do you want the rows _reordered_, or do you want each row's _position_ in the order as a new number (its rank)? Sorting and ranking look alike and answer different questions, and the answers differ most exactly where real data is awkward: ties and gaps.

This question builds the four tools you use for it: a stable multi-key sort, ranking with a chosen tie rule, a top-N that handles ties and missing values predictably, and a percentile rank that puts every value on a 0 to 1 scale.

### From theory to code

Implement `sort_by_columns(df, columns, ascending)`, `rank_scores(s, method)`, `top_n(df, column, n)` and `percent_rank(s)`. The signatures and docstrings are already in the editor.

### Constraints

- `sort_by_columns(df, columns, ascending)` sorts by the listed columns in order, each with its own direction (`ascending` is a list of bools the same length as `columns`). Rows that tie on every key keep their **original relative order** (a stable sort). Missing values go last in every key. The index labels travel with their rows.
- `rank_scores(s, method)` ranks so that the **highest** value gets rank `1`. `method` is `"min"`, `"dense"` or `"average"`: tied values get the lowest rank of the group, the next consecutive rank, or the mean of the group's ranks. Missing values stay `NaN`. The result is a float Series with `s`'s index.
- `top_n(df, column, n)` returns the `n` rows with the largest `column` values, largest first. Ties keep the original relative order. Rows where `column` is missing are never selected, so fewer than `n` rows can come back.
- `percent_rank(s)` returns `(rank_min - 1) / (count - 1)`, where `rank_min` ranks ascending (smallest value gets 1, ties share the lowest rank) and `count` is the number of non-missing values, which is at least 2. The smallest value maps to `0.0` and the largest to `1.0`. Missing stays `NaN`.

### Hints

<details>
<summary>Hint 1</summary>

`sort_values` accepts a list of columns and a list of directions. Its default algorithm is not stable for several keys; look at its `kind` argument.

</details>

<details>
<summary>Hint 2</summary>

`Series.rank` has `method` and `ascending` parameters. Flipping the direction is how the highest value becomes rank 1.

</details>

<details>
<summary>Hint 3</summary>

Drop the rows with a missing value in `column` first, then sort descending, then take the first `n`.

</details>
