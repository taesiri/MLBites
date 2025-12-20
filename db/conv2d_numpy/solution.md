## Approach

- **Understand the output dimensions**: Given input size, kernel size, stride, and padding, compute the output height and width using the standard formulas.
- **Apply zero-padding**: If padding > 0, pad the input with zeros around the spatial dimensions (H and W), leaving batch and channel dimensions unchanged.
- **Slide the kernel**: For each position in the output, extract the corresponding window from the padded input.
- **Compute dot products**: At each position, perform element-wise multiplication between the input window and each filter, sum across all input channels and spatial dimensions.
- **Add bias**: If provided, add the bias term to each output channel.
- **Use nested loops for clarity**: A straightforward 4-loop implementation (batch, output height, output width, output channel) is acceptable for an interview setting.

## Math

The 2D convolution operation for a single output position:

\[
\text{out}[n, c_{\text{out}}, h, w] = \text{bias}[c_{\text{out}}] + \sum_{c=0}^{C_{\text{in}}-1} \sum_{i=0}^{kH-1} \sum_{j=0}^{kW-1} \text{weight}[c_{\text{out}}, c, i, j] \cdot x_{\text{pad}}[n, c, h \cdot s + i, w \cdot s + j]
\]

Where:
- \(s\) is the stride
- \(x_{\text{pad}}\) is the zero-padded input
- The sums run over all input channels and kernel positions

Output dimensions:
\[
H_{\text{out}} = \frac{H + 2 \cdot \text{padding} - kH}{\text{stride}} + 1
\]
\[
W_{\text{out}} = \frac{W + 2 \cdot \text{padding} - kW}{\text{stride}} + 1
\]

## Correctness

- The implementation correctly handles the relationship between input windows and kernel positions.
- Zero-padding is applied symmetrically to all four sides of the spatial dimensions.
- The stride controls how much the window moves between adjacent output positions.
- Each output channel is computed independently using its corresponding filter.
- The bias is broadcast correctly to all spatial positions of each output channel.

## Complexity

- **Time**: \(O(N \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot kH \cdot kW)\) — for each output element, we sum over the entire receptive field.
- **Space**: \(O(N \cdot C_{\text{in}} \cdot (H + 2p) \cdot (W + 2p))\) for the padded input, plus \(O(N \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}})\) for the output.

## Common Pitfalls

- **Incorrect output size calculation**: Forgetting to account for padding or using wrong integer division.
- **Wrong padding axis**: Padding batch or channel dimensions instead of only height and width.
- **Off-by-one errors in indexing**: Incorrectly computing the window boundaries when extracting patches.
- **Forgetting stride**: Using `h` instead of `h * stride` when indexing into the padded input.
- **Summing over wrong axes**: Forgetting to sum over input channels when computing the output.
- **Bias shape mismatch**: Not properly broadcasting the bias to match the output spatial dimensions.



