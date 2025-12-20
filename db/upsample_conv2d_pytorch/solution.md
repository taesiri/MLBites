## Approach

- **Store upsampling config**: Save `scale_factor` and `mode` as instance attributes for use in the forward pass.
- **Create internal Conv2d**: Use PyTorch's `nn.Conv2d` with the specified `in_channels`, `out_channels`, `kernel_size`, and `padding`.
- **Two-step forward pass**:
  1. **Upsample**: Use `F.interpolate` to increase spatial dimensions by `scale_factor`
  2. **Convolve**: Apply the convolution to refine the upsampled features
- **Handle interpolation modes**: For `bilinear` mode, pass `align_corners=False` to get standard behavior; `nearest` mode doesn't require this argument.

## Math

The upsampling step scales the spatial dimensions:

\[
H_{up} = H_{in} \times \text{scale\_factor}, \quad W_{up} = W_{in} \times \text{scale\_factor}
\]

The convolution then produces output with dimensions:

\[
H_{out} = H_{up} + 2P - K + 1 = H_{in} \times \text{scale\_factor} + 2P - K + 1
\]

\[
W_{out} = W_{up} + 2P - K + 1 = W_{in} \times \text{scale\_factor} + 2P - K + 1
\]

Where:
- \(P\) is the padding
- \(K\) is the kernel size

**Common pattern**: To preserve the upsampled size after convolution, use `padding = (kernel_size - 1) // 2` for odd kernel sizes.

## Correctness

- The implementation correctly chains upsampling and convolution operations.
- Using `F.interpolate` handles both nearest and bilinear modes correctly.
- The `align_corners=False` setting for bilinear mode matches the expected standard behavior.
- The internal `Conv2d` is properly registered as a submodule, making weights trainable.

## Complexity

- **Time**: \(O(N \cdot C_{in} \cdot H_{up} \cdot W_{up})\) for upsampling (simple memory copy/interpolation) plus \(O(N \cdot C_{out} \cdot C_{in} \cdot K^2 \cdot H_{out} \cdot W_{out})\) for convolution.
- **Space**: \(O(N \cdot C_{in} \cdot H_{up} \cdot W_{up})\) for the intermediate upsampled tensor, plus \(O(C_{out} \cdot C_{in} \cdot K^2)\) for learnable parameters.

## Common Pitfalls

- Forgetting to set `align_corners=False` for bilinear mode (can cause inconsistent behavior).
- Not registering the `Conv2d` as a proper submodule (forgetting `self.conv = ...` syntax).
- Confusing this pattern with transposed convolution (`ConvTranspose2d`), which uses learned upsampling.
- Using `scale_factor` as a float when it should be an int (can cause shape mismatches).
- Forgetting that the output size depends on both the upsampling factor and the convolution parameters.
- Not understanding when to use this vs. transposed convolution — this pattern is preferred when avoiding checkerboard artifacts is important.



