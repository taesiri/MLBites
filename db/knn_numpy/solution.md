# K-Nearest Neighbors Classifier Solution

## Approach
- **Distance computation**: Compute pairwise squared Euclidean distances between test points and training points.
- Use the efficient broadcasting formula: \( \|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \cdot a \cdot b \).
- **Neighbor selection**: For each test point, find the indices of the k closest training points using `argsort`.
- **Majority voting**: Count the occurrences of each label among the k neighbors and return the most frequent one.
- Handle ties by returning the smallest label.

## Math

The squared Euclidean distance between test point \( x_i \) and training point \( t_j \) is:

\[
d_{ij}^2 = \|x_i - t_j\|^2 = \|x_i\|^2 + \|t_j\|^2 - 2 \, x_i \cdot t_j
\]

The predicted label for test point \( x_i \) is:

\[
\hat{y}_i = \arg\max_c \sum_{j \in N_k(x_i)} \mathbf{1}[y_j = c]
\]

where \( N_k(x_i) \) is the set of indices of the k nearest training points to \( x_i \).

## Correctness
- The distance formula avoids explicit loops and is numerically equivalent to `(X_test[:, None] - X_train) ** 2`.
- `argsort` along axis 1 gives the indices that would sort each row, so taking the first k gives the k nearest neighbors.
- `np.unique` with `return_counts=True` efficiently counts label occurrences.
- Tie-breaking by selecting the minimum label ensures deterministic output.

## Complexity
- **Time**:
  - Distance computation: \( O(n_{test} \cdot n_{train} \cdot d) \)
  - Sorting for neighbor selection: \( O(n_{test} \cdot n_{train} \log n_{train}) \)
  - Majority voting: \( O(n_{test} \cdot k \log k) \)
- **Space**:
  - Distance matrix: \( O(n_{test} \cdot n_{train}) \)
  - Neighbor indices: \( O(n_{test} \cdot k) \)

## Common Pitfalls
- Using slow nested loops instead of vectorized distance computation.
- Forgetting to handle ties in majority voting (returning arbitrary labels).
- Computing full Euclidean distance with `sqrt` (unnecessary for comparison).
- Off-by-one errors when slicing the k nearest neighbors.
- Returning wrong shape (e.g., `(n_test, 1)` instead of `(n_test,)`).
- Not sorting the distance matrix correctly (sorting along wrong axis).


