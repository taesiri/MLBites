# Solution: Implement SGD with Momentum

## Approach

1. **Store parameters**: Convert the input iterator to a list for repeated access.
2. **Initialize velocity buffers**: Create a zero tensor with the same shape as each parameter to store accumulated momentum.
3. **In `step()`, iterate over each parameter**:
   - Read the gradient from `p.grad`
   - Skip parameters with `None` gradients (frozen or unused)
   - Update velocity in-place: `v = momentum * v + g`
   - Update parameter in-place: `p = p - lr * v`
4. **Use `@torch.no_grad()`** to prevent tracking these operations in autograd.
5. **Use in-place tensor operations** (`mul_`, `add_`) for efficiency and to match PyTorch's API.

## Correctness

- The velocity buffer persists across steps, accumulating gradient history.
- On the first step with velocity initialized to zero: `v = 0 * μ + g = g`, so it behaves like vanilla SGD.
- On subsequent steps, velocity accumulates: directions with consistent gradients accelerate.
- Skipping `None` gradients handles frozen parameters correctly.
- Using `@torch.no_grad()` ensures optimizer updates don't create computation graphs.

## Complexity

- **Time**: O(N) per step, where N is the total number of parameters.
- **Space**: O(N) for velocity buffers (one buffer per parameter tensor).

## Common Pitfalls

1. **Forgetting to initialize velocity buffers** — velocity must persist across steps; recreating each step breaks momentum.
2. **Wrong update order** — velocity must be updated before using it to update parameters.
3. **Not handling `None` gradients** — some parameters may not have gradients (frozen layers, unused params).
4. **Modifying gradients instead of velocity** — the gradient tensor should not be modified; use a separate velocity buffer.
5. **Missing `@torch.no_grad()`** — without it, updates would be tracked by autograd, wasting memory and computation.
6. **Using wrong tensor ops** — `mul_` and `add_` are in-place; using `mul` and `add` would create new tensors and break updates.

