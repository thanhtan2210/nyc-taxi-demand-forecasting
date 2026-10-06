import os
import joblib


class DemandPredictor:
    """Loads the trained XGBoost demand model and runs inference on engineered features."""

    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.model = self._load_model()

        # Required input features, in training order
        self.features_cols = [
            'PULocation_Key', 'Hour', 'DayOfWeek', 'Is_Weekend', 'hour_sin', 'hour_cos',
            'lag_1h', 'lag_2h', 'lag_24h', 'lag_168h', 'rolling_mean_6h',
            'Temperature', 'Precipitation'
        ]

    def _load_model(self):
        """Retrieves the serialized model artifact from the 'saved_models/' directory."""
        path = os.path.join(self.root_dir, 'saved_models/xgboost/xgb_demand_model_v1.joblib')
        try:
            return joblib.load(path)
        except Exception as e:
            raise Exception(f"Initialization failure for the XGBoost model. Trace: {e}")

    def predict(self, raw_input_df):
        """Executes model inference on a DataFrame holding the feature columns."""
        return self.model.predict(raw_input_df[self.features_cols])
