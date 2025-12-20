## Approach

- **Store configuration**: Save in_channels, out_channels, kernel_size, stride, and padding as instance attributes.
- **Initialize learnable parameters**: Create weight tensor of shape `(out_channels, in_channels, kernel_size, kernel_size)` and bias of shape `(out_channels,)`.
- **Use proper initialization**: Apply Kaiming uniform initialization for weights and uniform initialization for bias (matching PyTorch defaults).
- **Apply padding**: If padding > 0, pad the input with zeros on all sides using `F.pad`.
- **Compute output dimensions**: Calculate output height and width based on the formula `(input_size + 2*padding - kernel_size) // stride + 1`.
- **Slide the kernel**: Iterate over all output positions, extract the corresponding input patch, and compute the weighted sum.
- **Use einsum for efficient computation**: For each output position, use `torch.einsum` to compute the dot product between the input patch and all filters simultaneously.

## Math

The 2D convolution operation computes:

\[
\text{output}[n, c_{out}, h, w] = \text{bias}[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \sum_{k_h=0}^{K-1} \sum_{k_w=0}^{K-1} \text{weight}[c_{out}, c_{in}, k_h, k_w] \cdot \text{input}[n, c_{in}, h \cdot s + k_h, w \cdot s + k_w]
\]

Where:
- \(n\) is the batch index
- \(c_{out}\) is the output channel index
- \(h, w\) are the output spatial coordinates
- \(C_{in}\) is the number of input channels
- \(K\) is the kernel size
- \(s\) is the stride

The output dimensions are:

\[
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{s} \right\rfloor + 1
\]

\[
W_{out} = \left\lfloor \frac{W_{in} + 2P - K}{s} \right\rfloor + 1
\]

Where \(P\) is the padding.

## Correctness

- The implementation correctly handles arbitrary batch sizes, channel counts, and spatial dimensions.
- Padding is applied symmetrically on all sides before convolution.
- The stride is applied correctly when determining the starting position of each patch.
- Using `einsum` ensures correct reduction over input channels and kernel dimensions.
- The bias is broadcast correctly to all batch elements and spatial positions.

## Complexity

- **Time**: \(O(N \cdot C_{out} \cdot C_{in} \cdot K^2 \cdot H_{out} \cdot W_{out})\) where N is batch size. Each output element requires summing over `in_channels * kernel_size^2` values.
- **Space**: \(O(C_{out} \cdot C_{in} \cdot K^2)\) for the learnable parameters, plus \(O(N \cdot C_{out} \cdot H_{out} \cdot W_{out})\) for the output.

## Common Pitfalls

- Forgetting to apply padding before computing output dimensions.
- Off-by-one errors when computing output height/width or patch extraction indices.
- Incorrect weight initialization (not matching PyTorch defaults).
- Forgetting to add the bias term.
- Using wrong dimension ordering — PyTorch uses NCHW (batch, channels, height, width).
- Not handling stride correctly when extracting patches.



