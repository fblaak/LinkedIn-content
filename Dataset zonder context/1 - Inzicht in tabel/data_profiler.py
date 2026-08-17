from pathlib import Path

import pandas as pd
import numpy as np

class DataProfiler:
    """Create a simple column profile for a pandas DataFrame."""

    def __init__(self, df, table_name="dataset", example_count=3):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(table_name, str):
            raise TypeError("table_name must be a string.")
        if not isinstance(example_count, int) or example_count < 1:
            raise ValueError("example_count must be a positive integer.")

        self.df = df
        self.table_name = table_name
        self.example_count = example_count
        self.row_count = len(df)

    @staticmethod
    def _non_empty(series):
        """Exclude NULLs, empty strings and whitespace-only strings."""
        values = series.dropna()

        if (
            pd.api.types.is_object_dtype(values)
            or pd.api.types.is_string_dtype(values)
        ):
            values = values[values.astype(str).str.strip().ne("")]

        return values

    @staticmethod
    def _unique_count(values):
        """Count unique values, also for unusual object columns."""
        try:
            return int(values.nunique())
        except TypeError:
            return int(values.astype(str).nunique())

    def _profile_column(self, column_name):
        series = self.df[column_name]
        values = self._non_empty(series)

        non_empty_count = len(values)
        empty_count = self.row_count - non_empty_count
        unique_count = self._unique_count(values)

        unique_percentage = (
            unique_count / non_empty_count * 100
            if non_empty_count
            else np.nan
        )

        empty_percentage = (
            empty_count / self.row_count * 100
            if self.row_count
            else np.nan
        )

        try:
            examples = (
                values
                .drop_duplicates()
                .head(self.example_count)
                .tolist()
            )
        except TypeError:
            examples = (
                values
                .astype(str)
                .drop_duplicates()
                .head(self.example_count)
                .tolist()
            )

        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )

        return {
            "table_name": self.table_name,
            "column_name": column_name,
            "data_type": str(series.dtype),
            "examples": examples,
            "row_#": self.row_count,
 
            "empty_#": empty_count,
            "empty_%": round(empty_percentage, 2),

            "unique_#": unique_count,
            "unique_%": round(unique_percentage, 2),
            "is_unique": non_empty_count > 0 and unique_count == non_empty_count,

            "min": series.min(skipna=True) if is_numeric else None,
            "mean": series.mean(skipna=True) if is_numeric else None,
            "max": series.max(skipna=True) if is_numeric else None,
        }

    def profile(self):
        """Profile all columns."""
        results = []

        for column_name in self.df.columns:
            try:
                results.append(self._profile_column(column_name))
            except Exception as error:
                print(
                    f"Warning: '{self.table_name}.{column_name}' "
                    f"could not be profiled: {error}"
                )

        return pd.DataFrame(results)


def profile_dataframe(df, table_name="dataset", example_count=3):
    """Profile one DataFrame."""
    return DataProfiler(
        df=df,
        table_name=table_name,
        example_count=example_count,
    ).profile()


def profile_folder(folder_path, example_count=3):
    """Profile all CSV files in a folder and combine the results."""
    files = list(Path(folder_path).glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{folder_path}'."
        )

    profiles = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            profiles.append(
                profile_dataframe(
                    df=df,
                    table_name=file_path.stem,
                    example_count=example_count,
                )
            )

        except Exception as error:
            print(f"Warning: '{file_path.name}' could not be read: {error}")

    if not profiles:
        raise ValueError("None of the CSV files could be profiled.")

    return pd.concat(profiles, ignore_index=True)