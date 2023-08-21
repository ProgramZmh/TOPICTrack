import numpy as np

def set_non_min_to_one_by_row(arr):
    # Find the minimum value in each row
    min_values = np.min(arr, axis=1, keepdims=True)
    
    # Create a mask for non-minimum values in each row
    mask = arr > min_values
    
    # Set non-minimum values to 1
    arr[mask] = 1

    return arr

# Example usage:
arr = np.array([[3, 2, 4],
                [1, 5, 2],
                [6, 3, 7]])

result = set_non_min_to_one_by_row(arr)
print(result)
