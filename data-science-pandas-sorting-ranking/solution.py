import pandas as pd


def sort_by_columns(df: pd.DataFrame, columns: list, ascending: list) -> pd.DataFrame:
    """
    Sort by `columns` in order, each with its own direction from `ascending`.
    Ties on every key keep their original order. NaN goes last. Index labels
    travel with their rows.
    """
    # TODO: Multi-key stable sort.
    return df.sort_values(columns , ascending = ascending)


def rank_scores(s: pd.Series, method: str) -> pd.Series:
    """
    Highest value = rank 1. method is "min", "dense" or "average". NaN stays
    NaN. Float Series with s's index.
    """
    # TODO: Rank from the top.
    return s.rank(method = method , ascending = False).astype(float)
    


def top_n(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    """
    The n rows with the largest `column`, largest first; ties keep their
    original order; rows with a missing `column` are never chosen.
    """
    # TODO: Drop missing, sort descending, take n.
    return df.dropna(subset = [column] ).sort_values(
        by= column , ascending = False , kind = 'stable'
    ).head(n)
    


def percent_rank(s: pd.Series) -> pd.Series:
    """
    (rank_min - 1) / (count - 1) with rank_min ascending (smallest = 1). The
    smallest value is 0.0, the largest 1.0. NaN stays NaN.
    """
    # TODO: Rescale the ascending min-ranks to 0..1.

    count = s.count()
    ranks = s.rank(method = 'min' , ascending= True)
    if count <= 1 :
        return s.astype(float) * 0.0
    return (ranks - 1)/ (count -1)
