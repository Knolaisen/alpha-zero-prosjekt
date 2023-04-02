import numpy as np


def softmax(vector: np.array) -> np.ndarray:
    """
    Compute softmax values for each sets of scores in x.

    """
    return np.exp(vector) / np.sum(np.exp(vector), axis=0)


if __name__ == "__main__":
    # Example usage:
    input_vector = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    output_vector = softmax(input_vector)
    print("Input vector:", input_vector)
    print("Output vector:", output_vector)
