import numpy as np


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


if __name__ == "__main__":
    vector_a = [1, 0]
    vector_b = [1, 0]
    vector_c = [0, 1]

    print("Same direction:", cosine_similarity(vector_a, vector_b))
    print("Perpendicular:", cosine_similarity(vector_a, vector_c))
