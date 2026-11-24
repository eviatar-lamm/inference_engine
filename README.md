# inference_engine

A C++23 inference engine in progress, with GPT-2 as the first target. The current executable runs a tensor demo.

## Implemented so far

- **Tensor storage and views.** `Storage` owns the data. `Tensor` uses a shape, strides, and an offset to index that data. A view shares the original buffer, so slicing avoids copying elements and writes through a view update the parent.
- **Tensor operations.** The FP32 path supports indexing, iteration over dense tensors and strided views, scalar tensors, and element-wise addition and multiplication with tensors or scalars. `operator[]` takes one `Slice` per supplied axis, uses inclusive stops for positive steps, and supports slicing an existing view.
- **Model weights.** `SafeTensor` reads the file header and loads tensors with their shapes and dtypes. `InferenceModel` stores the tensors by name. A local GPT-2 checkpoint loaded 160 tensors. [`install_model.py`](install_model.py) downloads the checkpoint, config, and tokenizer files into `models/`.
- **Matrix multiplication.** There is a first blocked FP32 implementation for dense 2D tensors.

For example, this slice keeps both rows and selects columns `0` and `2`. The view has shape `{2, 2}` and strides `{4, 2}` in the original buffer.

```cpp
Tensor tensor({2, 4});
auto view = tensor[Slice{}, Slice{0, 3, 2}];
```

## What's left

1. **Finish tensor edge cases.** The current suite passes 13 of 17 tests. The four failures cover out-of-bounds indexing, zero slice steps, iteration over a zero-length axis, and element-wise operations on mismatched shapes with the same element count.
2. **Finish matrix multiplication.** Fix the output column loop for rectangular matrices and add tests for square and rectangular inputs.
3. **Finish weight loading.** Make `get_layer()` look up the requested name, validate headers and tensor reads, and handle dtypes explicitly. Tensor arithmetic currently runs on FP32.
4. **Run GPT-2 end to end.** Implement the forward pass, connect the tokenizer and generation loop, and compare the model's outputs with a reference implementation.

## Build and test

Use CMake and a compiler with C++23 multidimensional subscript support.

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
