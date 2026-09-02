import numpy as np

# Define the Rotation Matrix R as a 2D NumPy array
R = np.array([
    [0.9882, 0.0760, 0.1328],
    [0.0000, 0.8678, -0.4970],
    [-0.1530, 0.4911, 0.8576]
])

# Option 1: Simple Print
print("Rotation Matrix R:")
print(R)

# Option 2: Formatted Print (Looks like a true matrix)
print("\nFormatted Matrix R:")
for row in R:
    print(f"| {row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f} |")