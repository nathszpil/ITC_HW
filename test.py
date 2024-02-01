import pandas as pd
import numpy as np

# Create a vector of 594017 numbers with the same value (e.g., 42)
pred_vector = np.full(594017, 42)

# Save the vector to a CSV file
pd.Series(pred_vector).to_csv('my_submission.txt', sep=',', lineterminator='\n', index=False, float_format='%.8f',
                              header=None, encoding='ascii')
