# test.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class MovieFeatureEngineeringPipeline:
    def __init__(self, 
                feature_scaler: Any,
                transform_data: Dict[str, Any],
                actor_data: Dict[str, Dict],
                director_data: Dict[str, Dict],
                producer_data: Dict[str, Dict],
                writer_data: Dict[str, Dict]):
        
        self.feature_scaler = feature_scaler
        self.transform_data = transform_data
        self.actor_data = actor_data
        self.director_data = director_data
        self.producer_data = producer_data
        self.writer_data = writer_data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying the input
        df = df.copy()
        
        # Add metrics for each role
        df = self.add_actor_metrics(df)
        df = self.add_director_metrics(df)
        df = self.add_producer_metrics(df)
        df = self.add_writer_metrics(df)
        
        # Add combined and derived features
        df = self.add_combined_metrics(df)
        df = self.add_budget_features(df)
        
        # Scale the engineered features
        df = self.scale_features(df)
        
        return df

    def add_actor_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Copy the implementation from your original code
        metrics = self.actor_data['metrics']
        
        df['cast_avg_revenue'] = 0.0
        df['cast_revenue_consistency'] = 0.0
        df['cast_hit_rate'] = 0.0
        df['cast_popularity'] = 0.0
        df['cast_pop_consistency'] = 0.0
        df['cast_rev_pop_correlation'] = 0.0
        df['cast_experience'] = 0.0
        df['top_cast_composite_score'] = 0.0
        
        cast_columns = [col for col in df.columns if col.startswith('cast_') and col in self.actor_data['columns']]
        
        for idx, row in df.iterrows():
            present_actors = [col.replace('cast_', '') for col in cast_columns if row[col] == 1]
            
            if present_actors:
                actor_metrics = [metrics[actor] for actor in present_actors if actor in metrics]
                if actor_metrics:
                    df.loc[idx, 'cast_avg_revenue'] = np.mean([m['avg_revenue'] for m in actor_metrics])
                    df.loc[idx, 'cast_revenue_consistency'] = np.mean([m['revenue_consistency'] for m in actor_metrics])
                    df.loc[idx, 'cast_hit_rate'] = np.mean([m['hit_rate'] for m in actor_metrics])
                    df.loc[idx, 'cast_popularity'] = np.mean([m['avg_popularity'] for m in actor_metrics])
                    df.loc[idx, 'cast_pop_consistency'] = np.mean([m['popularity_consistency'] for m in actor_metrics])
                    df.loc[idx, 'cast_rev_pop_correlation'] = np.mean([m['revenue_popularity_correlation'] for m in actor_metrics])
                    df.loc[idx, 'cast_experience'] = np.mean([m['movies_count'] for m in actor_metrics])
                    df.loc[idx, 'top_cast_composite_score'] = np.mean([m['composite_score'] for m in actor_metrics])
        
        return df

    # Add all other methods from your original code...
    def add_director_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass

    def add_producer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass

    def add_writer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass

    def add_combined_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass

    def add_budget_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation from your original code
        pass