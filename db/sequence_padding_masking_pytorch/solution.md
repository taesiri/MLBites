## Approach

### `pad_sequences`
- Compute the length of each input sequence and store in a tensor.
- Find the maximum length to determine the output shape.
- Create an output tensor of shape `(B, max_len)` filled with `pad_value`.
- Copy each sequence into its corresponding row of the output tensor.

### `create_padding_mask`
- Create a 1D tensor of position indices `[0, 1, ..., max_len-1]`.
- Use broadcasting to compare positions against lengths: `positions >= lengths[:, None]`.
- Positions that are at or beyond the sequence length are marked as `True` (padding).

### `create_causal_mask`
- Create row and column index tensors for a `(T, T)` matrix.
- Compare column indices against row indices: `col > row`.
- Positions where `j > i` are future positions and should be masked (`True`).

## Math

The padding mask for a sequence of length \(L\) in a padded tensor of length \(T\) is:

\[
\text{mask}[i] = \begin{cases} \text{False} & \text{if } i < L \\ \text{True} & \text{if } i \geq L \end{cases}
\]

The causal mask for autoregressive attention is:

\[
\text{mask}[i, j] = \begin{cases} \text{False} & \text{if } j \leq i \\ \text{True} & \text{if } j > i \end{cases}
\]

When applied to attention scores, masked positions (where mask is `True`) are set to \(-\infty\) before softmax, resulting in zero attention weight.

## Correctness
- `pad_sequences` preserves original sequence values in the left positions and fills remaining positions with the pad value.
- `create_padding_mask` correctly identifies positions beyond each sequence's actual length using vectorized comparison.
- `create_causal_mask` creates a proper upper-triangular mask (excluding diagonal) that blocks attention to future positions.

## Complexity
- **`pad_sequences`**: \(O(B \cdot T)\) time and space, where \(B\) is batch size and \(T\) is max sequence length.
- **`create_padding_mask`**: \(O(B \cdot T)\) time and space.
- **`create_causal_mask`**: \(O(T^2)\) time and space.

## Common Pitfalls
- Confusing mask semantics: in PyTorch attention, `True` typically means "mask out" (ignore), not "attend to".
- Off-by-one errors in the causal mask: position `i` should attend to positions `0..i` inclusive.
- Forgetting to handle empty sequences or edge cases like `seq_len=0`.
- Not preserving the device/dtype of input tensors in the output.
- Using the wrong comparison operator (`>` vs `>=`) when creating masks.



