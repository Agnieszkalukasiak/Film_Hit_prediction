import os
import joblib
import pandas as pd

def resave_pickles(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                filepath = os.path.join(root, file)
                try:
                    # Load with any protocol
                    data = pd.read_pickle(filepath)
                    # Resave with protocol 4
                    if isinstance(data, pd.DataFrame):
                        data.to_pickle(filepath, protocol=4)
                    else:
                        joblib.dump(data, filepath, protocol=4)
                    print(f"Resaved: {filepath}")
                except Exception as e:
                    print(f"Error with {filepath}: {e}")

# Run from project root
resave_pickles('/workspace/Film_Hit_prediction')