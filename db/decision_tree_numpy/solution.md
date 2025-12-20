# Decision Tree Classifier Solution

## Approach
- **Gini impurity**: Measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the class distribution.
- Computed as \(1 - \sum p_i^2\) where \(p_i\) is the proportion of class \(i\).
- A pure node (all same class) has Gini = 0; maximum impurity occurs with uniform class distribution.
- **Finding best split**: For each feature, try each unique value as a potential threshold.
- Split rule: left child gets samples where `X[:, feature] <= threshold`.
- Compute weighted average of child Gini impurities; pick the split with lowest weighted Gini.
- Empty children are invalid splits and are skipped.

## Math

Gini impurity for a set of labels with class probabilities \(p_1, p_2, \ldots, p_C\):

\[
G = 1 - \sum_{i=1}^{C} p_i^2
\]

For a split that divides \(n\) samples into left (\(n_L\)) and right (\(n_R\)) subsets:

\[
G_{\text{weighted}} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R
\]

The best split minimizes \(G_{\text{weighted}}\).

## Correctness
- Gini impurity correctly measures the probability of misclassification under random labeling.
- Trying all unique feature values ensures we find the optimal threshold for each feature.
- Weighted average accounts for different child sizes, preventing bias toward unbalanced splits.
- Edge cases (empty children, all same values) are handled by skipping invalid splits.

## Complexity
- **compute_gini**:
  - Time: \(O(n)\) for counting, \(O(C)\) for computing probabilities where \(C\) is number of classes
  - Space: \(O(C)\) for storing counts
- **find_best_split**:
  - Time: \(O(n \cdot d \cdot n) = O(n^2 d)\) where \(n\) is samples, \(d\) is features
    - For each feature: \(O(n \log n)\) to sort/unique, \(O(n^2)\) worst case for all thresholds
  - Space: \(O(n)\) for masks and unique values

## Common Pitfalls
- Forgetting to weight the child Gini values by their sizes.
- Using `<` instead of `<=` for the split condition (must be consistent with threshold interpretation).
- Not handling the case where a threshold creates an empty child (e.g., all values equal).
- Computing Gini as sum of probabilities instead of sum of squared probabilities.
- Returning wrong types or shapes (feature index must be int, not float).
- Off-by-one errors when the threshold equals the maximum feature value (right child becomes empty).




