# K-Means Clustering Solution

## Approach
- **Assignment step**: For each point, compute its squared Euclidean distance to all centroids and pick the nearest one.
- Use broadcasting to compute pairwise distances efficiently: \( \|x - c\|^2 = \|x\|^2 + \|c\|^2 - 2 x \cdot c \).
- The cluster assignment is the `argmin` over centroids for each point.
- **Update step**: For each cluster, compute the mean of all points assigned to it.
- Handle empty clusters by setting their centroid to the origin (zeros).
- Both operations are vectorized for efficiency.

## Math

The squared Euclidean distance between point \( x_i \) and centroid \( c_j \) is:

\[
d_{ij}^2 = \|x_i - c_j\|^2 = \|x_i\|^2 + \|c_j\|^2 - 2 \, x_i \cdot c_j
\]

Assignment:
\[
a_i = \arg\min_j \, d_{ij}^2
\]

Centroid update:
\[
c_j = \frac{1}{|S_j|} \sum_{i \in S_j} x_i
\]

where \( S_j = \{i : a_i = j\} \) is the set of points assigned to cluster \( j \).

## Correctness
- The distance formula is algebraically equivalent to computing `(X - centroids)**2` but avoids explicit loops.
- `argmin` along the centroid axis gives the nearest cluster for each point.
- The mean over assigned points is the correct centroid update per Lloyd's algorithm.
- Empty clusters are handled gracefully by returning zeros.

## Complexity
- **assign_clusters**:
  - Time: \( O(n \cdot k \cdot d) \) for distance computation, \( O(n \cdot k) \) for argmin
  - Space: \( O(n \cdot k) \) for the distance matrix
- **update_centroids**:
  - Time: \( O(n \cdot d) \) to compute means (linear scan through data)
  - Space: \( O(k \cdot d) \) for the output centroids

## Common Pitfalls
- Using slow nested loops instead of vectorized distance computation.
- Forgetting to handle empty clusters (division by zero).
- Computing full Euclidean distance instead of squared distance (unnecessary sqrt).
- Off-by-one errors in cluster indexing.
- Returning wrong shapes (e.g., `(n_samples, 1)` instead of `(n_samples,)` for assignments).
