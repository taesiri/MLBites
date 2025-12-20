## Mathematical Background

A vanilla RNN (also known as Elman RNN) processes sequential data by maintaining a hidden state that captures information from previous timesteps.

### Recurrence Relation

At each timestep \( t \), the hidden state is computed as:

\[
h_t = \tanh(W_{ih} \cdot x_t + b_{ih} + W_{hh} \cdot h_{t-1} + b_{hh})
\]

Where:
- \( x_t \in \mathbb{R}^{input\_size} \): input at timestep \( t \)
- \( h_{t-1} \in \mathbb{R}^{hidden\_size} \): hidden state from previous timestep
- \( W_{ih} \in \mathbb{R}^{hidden\_size \times input\_size} \): input-to-hidden weights
- \( W_{hh} \in \mathbb{R}^{hidden\_size \times hidden\_size} \): hidden-to-hidden weights
- \( b_{ih}, b_{hh} \in \mathbb{R}^{hidden\_size} \): biases

### Weight Initialization

PyTorch initializes RNN weights uniformly in the range \([-k, k]\) where:
\[
k = \frac{1}{\sqrt{hidden\_size}}
\]

---

## Approach

- Store `input_size` and `hidden_size` for later use.
- Initialize four learnable parameters (`W_ih`, `W_hh`, `b_ih`, `b_hh`) using uniform distribution in `[-k, k]`.
- In `__call__`:
  - If `h0` is `None`, initialize hidden state to zeros.
  - Squeeze `h0` from shape `(1, batch, hidden_size)` to `(batch, hidden_size)`.
  - Loop through each timestep `t`:
    - Extract `x_t = x[t]` of shape `(batch, input_size)`.
    - Compute `h = tanh(x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh)`.
    - Append `h` to outputs list.
  - Stack outputs to get shape `(seq_len, batch, hidden_size)`.
  - Unsqueeze final `h` to get `h_n` of shape `(1, batch, hidden_size)`.

## Correctness

- The recurrence formula matches the standard Elman RNN definition used by `torch.nn.RNN`.
- Weight initialization follows PyTorch's default strategy, ensuring similar behavior.
- The final hidden state `h_n` equals `output[-1:]`, which is the expected behavior.
- Gradients flow correctly through all operations since we use standard PyTorch tensor ops.

## Complexity

- **Time:** \( O(seq\_len \times batch \times hidden\_size \times (input\_size + hidden\_size)) \) — one matrix multiply per timestep.
- **Space:** \( O(seq\_len \times batch \times hidden\_size) \) for storing all hidden states.

## Common Pitfalls

- Forgetting to handle `h0=None` case (should default to zeros).
- Using wrong matrix dimensions (remember: `W @ x.T` vs `x @ W.T`).
- Not using `unsqueeze(0)` for the final hidden state (should have leading dim of 1).
- Forgetting to use `tanh` activation (leaving it linear).
- Initializing weights with wrong distribution (should match PyTorch's uniform `[-k, k]`).
- Processing sequence in wrong order (should iterate from `t=0` to `t=seq_len-1`).




