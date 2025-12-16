Approach
- Iterate over all parameters and read gradients from `p.grad`.
- Optionally apply weight decay by adding `weight_decay * p` to the gradient.
- Optionally maintain a momentum buffer per parameter:
  - `buf = momentum * buf + d_p`
  - then use `d_p = buf`
- Update parameters in-place: `p -= lr * d_p`.

Correctness
- Weight decay matches the standard L2 regularization form used by `torch.optim.SGD`.
- Momentum buffers accumulate past gradients with exponential decay, matching SGD momentum behavior.
- Parameters with `grad is None` are skipped, consistent with PyTorch optimizers.

Complexity
- Time: \(O(\sum_i |p_i|)\) per step.
- Space: \(O(\sum_i |p_i|)\) for momentum buffers when momentum is enabled.

Common Pitfalls
- Forgetting to apply weight decay to the gradient (when requested).
- Updating momentum buffers with the wrong tensor (use the current `d_p`).
- Performing updates without `torch.no_grad()` (accidentally tracking optimizer ops).


