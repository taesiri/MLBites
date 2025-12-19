# Solution: Implement GRU from Scratch

## Approach
- Create linear layers for input-to-hidden and hidden-to-hidden projections. Unlike LSTM which has 4 gates, GRU has 3 gates (r, z, n) but the new gate (n) applies the reset gate before the hidden projection.
- For reset (r) and update (z) gates: use combined linear layers projecting to `2 * hidden_size`.
- For new gate (n): use separate linear layers since the reset gate is applied between the hidden projection.
- Initialize hidden state `h_t` to zeros if not provided.
- Loop through each timestep:
  1. Compute reset and update gates: `r, z = sigmoid(W_rz @ x_t + W_rz_h @ h_t)`
  2. Compute new gate: `n = tanh(W_n @ x_t + r * (W_n_h @ h_t))`
  3. Update hidden state: `h_t = (1 - z) * n + z * h_{t-1}`
- Collect hidden states and stack into output tensor.

## Math

The GRU equations at each timestep:

\[r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})\]
\[z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})\]
\[n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))\]
\[h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}\]

Where:
- \(r_t\) is the reset gate (controls how much of the past to forget)
- \(z_t\) is the update gate (controls how much of the past to keep)
- \(n_t\) is the new/candidate gate (proposed new hidden state)
- \(\odot\) denotes element-wise multiplication

## Correctness
- The reset gate `r` controls how much of the previous hidden state to use when computing the candidate.
- The update gate `z` interpolates between the previous hidden state and the candidate.
- When `z = 1`, `h_t = h_{t-1}` (copy gate / skip connection).
- When `z = 0`, `h_t = n` (full update with new content).
- The final hidden state `h_n` equals `output[:, -1, :]`, ensuring consistency.

## Complexity
- Time: \(O(B \cdot T \cdot (input\_size \cdot hidden\_size + hidden\_size^2))\) for linear projections at each timestep.
- Space: \(O(B \cdot T \cdot hidden\_size)\) to store all hidden states in the output.

## Common Pitfalls
- Forgetting to apply the reset gate *before* computing `W_hn @ h_t` in the new gate computation.
- Confusing the update formula: it's `(1 - z) * n + z * h` not `z * n + (1 - z) * h`.
- Forgetting to initialize `h_t` with the correct dtype and device (should match input `x`).
- Using in-place operations on `h_t` which can break gradient computation.
- Incorrect chunking order when splitting gates—PyTorch's built-in GRU uses (r, z) ordering.


