# Solution: Implement LSTM from Scratch

## Approach
- Create two linear layers: one for input-to-hidden projection (`input_linear`) and one for hidden-to-hidden projection (`hidden_linear`), each projecting to `4 * hidden_size` to compute all four gates (i, f, g, o) in one pass.
- Initialize hidden state `h_t` and cell state `c_t` to zeros if not provided.
- Loop through each timestep, computing gates as: `gates = input_linear(x_t) + hidden_linear(h_t)`.
- Split gates into four chunks: input gate (i), forget gate (f), cell gate/candidate (g), and output gate (o).
- Apply sigmoid to i, f, o and tanh to g.
- Update cell state: `c_t = f * c_t + i * g` (forget old, add new).
- Update hidden state: `h_t = o * tanh(c_t)`.
- Collect hidden states at each timestep and stack them into the output tensor.

## Correctness
- The forget gate controls how much of the previous cell state to retain.
- The input gate controls how much of the new candidate to add.
- The output gate controls how much of the cell state is exposed as the hidden state.
- By using `tanh` for the candidate and cell output, values are bounded in `[-1, 1]`.
- Sigmoid gates produce values in `[0, 1]`, acting as soft switches.
- The final hidden state `h_n` equals `output[:, -1, :]`, ensuring consistency.

## Complexity
- Time: \(O(B \cdot T \cdot (input\_size \cdot hidden\_size + hidden\_size^2))\) for the linear projections at each timestep.
- Space: \(O(B \cdot T \cdot hidden\_size)\) to store all hidden states in the output.

## Common Pitfalls
- Forgetting to initialize `h_t` and `c_t` with the correct dtype and device (should match input `x`).
- Incorrect gate order when chunking—PyTorch's built-in LSTM uses (i, f, g, o) ordering.
- Applying tanh instead of sigmoid to the gates (or vice versa).
- Not returning both the sequence of hidden states and the final (h_n, c_n) tuple.
- Using in-place operations on `c_t` or `h_t` which can break gradient computation.

