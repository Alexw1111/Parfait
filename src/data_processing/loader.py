import pandas as pd
import numpy as np
from typing import List, Generator, Tuple

def load_stock_data(file_path: str, features: List[str]) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in {file_path}")
    df_processed = df[['date'] + features].copy()
    if 'volume' in df_processed.columns:
        df_processed['volume'] = np.log1p(df_processed['volume'])
    df_processed = df_processed.ffill().bfill()
    return df_processed

def create_sliding_windows(
    df: pd.DataFrame, 
    history_window: int, 
    future_horizon: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    total_len = len(df)
    for i in range(total_len - history_window - future_horizon + 1):
        history_end = i + history_window
        future_end = history_end + future_horizon
        
        history_df = df.iloc[i:history_end]
        future_df = df.iloc[history_end:future_end]
        
        yield history_df, future_df