import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from src.preprocessing.columns import *

class DataNormalization:
    def __init__(self, training_data, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            scores_to_norm = np.hstack([
                training_data[NORM_COLUMNS].astype(np.float64).values,
                self._clip_and_log_transform(training_data[LOG_NORM_COLUMNS])
            ])
            self.scaler.fit(scores_to_norm)

    def _clip_and_log_transform(self, data, log_gamma=0.1):
        return np.log(log_gamma + np.clip(data, 0, None))

    def _preprocess_normalized_data(self, df):
        df = df.copy()
        df[pd.isna(df)] = 0
        return df

    def transform(self, data):
        no_norm_scores = data[AS_IS_COLUMNS].astype(np.float64).values - 0.5
        scores_to_norm = np.hstack([
            data[NORM_COLUMNS].astype(np.float64).values,
            self._clip_and_log_transform(data[LOG_NORM_COLUMNS])
        ])
        normed = self.scaler.transform(scores_to_norm)

        MIMICzs = pd.DataFrame(
            np.hstack([no_norm_scores, normed]),
            columns=ALL_FEATURE_COLUMNS,
            index=data.index
        )
        return self._preprocess_normalized_data(MIMICzs)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump({
                'type': 'DataNormalization',
                'scaler': self.scaler
            }, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        assert obj['type'] == 'DataNormalization'
        return DataNormalization(None, obj['scaler'])