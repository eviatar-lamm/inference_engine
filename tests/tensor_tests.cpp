#include "tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Operation>
void check_throws(Operation operation, std::string_view message) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

Tensor make_tensor(std::vector<std::size_t> shape, std::initializer_list<float> values) {
    Tensor tensor(shape);
    std::copy(values.begin(), values.end(), tensor.get_raw<float>());
    return tensor;
}

void check_values(Tensor tensor, std::initializer_list<float> expected) {
    auto it = tensor.begin<float>();
    auto end = tensor.end<float>();
    for (float value : expected) {
        check(it != end, "iterator ended early");
        const float actual = *it;
        check(actual == value, "expected " + std::to_string(value) + ", got " + std::to_string(actual));
        ++it;
    }
    check(it == end, "iterator has more elements than expected");
}

void dense_layout() {
    auto tensor = make_tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    check(tensor.shape == std::vector<std::size_t>{2, 2, 3}, "shape differs from constructor input");
    check(tensor.strides == std::vector<std::size_t>{6, 3, 1}, "wrong row-major strides");
    check(tensor({1, 0, 2}).item<float>() == 9, "wrong value at [1, 0, 2]");
    tensor({0, 1, 0}).item<float>() = 20;
    check_values(tensor, {1, 2, 3, 20, 5, 6, 7, 8, 9, 10, 11, 12});
}

void dense_arithmetic() {
    auto left = make_tensor({2, 2}, {1, -2, 3, 4});
    auto right = make_tensor({2, 2}, {5, 6, -7, 8});
    check_values(left + right, {6, 4, -4, 12});
    check_values(left * right, {5, -12, -21, 32});
    check_values(left + 2, {3, 0, 5, 6});
    check_values(left * 0.5f, {0.5f, -1, 1.5f, 2});
    check_values(left, {1, -2, 3, 4});
}

void allocated_size() {
    Tensor tensor({2, 3});
    check(tensor.size() == 6, "a 2x3 tensor must contain six elements");
}

void view_size() {
    Tensor tensor({3, 3});
    auto row = tensor({1});
    check(row.size() == 3, "a row view must contain three elements");
}

void view_indexing() {
    auto tensor = make_tensor({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto row = tensor({1});
    check(row({2}).item<float>() == 6, "row[2] must refer to tensor[1, 2]");
    row({2}).item<float>() = 99;
    check(tensor({1, 2}).item<float>() == 99, "writing through a row view must update its parent");
}

void view_iteration() {
    auto tensor = make_tensor({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    check_values(tensor({1}), {4, 5, 6});
}

void view_arithmetic() {
    auto tensor = make_tensor({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto row = tensor({1});
    auto other = make_tensor({3}, {10, 20, 30});
    check_values(row + other, {14, 25, 36});
    check_values(row * other, {40, 100, 180});
}

void stepped_slice() {
    auto tensor = make_tensor({6}, {1, 2, 3, 4, 5, 6});
    auto view = tensor[Slice{0, 5, 2}];
    check(view.shape == std::vector<std::size_t>{3}, "a step of two over six elements must select three");
    check(view.strides == std::vector<std::size_t>{2}, "a step of two must double the stride");
    check(view.offset == 0, "a slice starting at zero must keep the origin");
    check_values(view, {1, 3, 5});
}

void stepped_slice_remainder() {
    Tensor tensor({5});
    auto view = tensor[Slice{0, 4, 2}];
    check(view.shape == std::vector<std::size_t>{3}, "the slice must include index 4 when the step lands on the stop");
}

void inner_axis_slice() {
    auto tensor = make_tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto view = tensor[Slice{}, Slice{0, 3, 2}];
    check(view.shape == std::vector<std::size_t>{2, 2}, "slicing columns must preserve both rows");
    check(view.strides == std::vector<std::size_t>{4, 2}, "column slice must retain the row stride");
    check(view.offset == 0, "full rows and a column start of zero must keep the origin");
    check_values(view, {1, 3, 5, 7});
}

void multi_axis_slice() {
    auto tensor = make_tensor({2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto view = tensor[Slice{1}, Slice{1, 2}];
    check(view.shape == std::vector<std::size_t>{1, 2, 4}, "omitting the last axis must keep all its elements");
    check(view.strides == std::vector<std::size_t>{12, 4, 1}, "unit steps must keep the original strides");
    check(view.offset == 16, "the starts on the first two axes must both contribute to the offset");
    check_values(view, {17, 18, 19, 20, 21, 22, 23, 24});
    *view.begin<float>() = 99;
    check(tensor({1, 1, 0}).item<float>() == 99, "writing through the slice must update its parent");
}

void nested_slice() {
    auto tensor = make_tensor({8}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto first = tensor[Slice{1, 6, 2}];
    auto second = first[Slice{1, 2}];
    check(second.offset == 3, "a nested slice must compose both offsets and the parent stride");
    check(second.strides == std::vector<std::size_t>{2}, "nested slice must retain the parent stride");
    check_values(second, {4, 6});
}

void out_of_bounds_index() {
    Tensor tensor({2, 3});
    check_throws<std::out_of_range>([&] { tensor({2, 0}); }, "out-of-bounds indexing must throw");
}

void zero_slice_step() {
    Tensor tensor({4});
    check_throws<std::invalid_argument>([&] { tensor[Slice{0, 3, 0}]; }, "a zero slice step must throw");
}

void empty_tensor() {
    Tensor tensor({2, 0, 3});
    check_values(tensor, {});
}

void scalar_tensor() {
    Tensor scalar({});
    scalar.item<float>() = 7;
    check_values(scalar, {7});
}

void incompatible_shapes() {
    auto left = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto right = make_tensor({3, 2}, {1, 2, 3, 4, 5, 6});
    check_throws<std::invalid_argument>([&] { left + right; }, "addition must reject incompatible shapes");
    check_throws<std::invalid_argument>([&] { left * right; }, "multiplication must reject incompatible shapes");
}

struct Test {
    std::string_view name;
    void (*run)();
};

constexpr Test tests[] = {
    {"dense_layout", dense_layout},
    {"dense_arithmetic", dense_arithmetic},
    {"allocated_size", allocated_size},
    {"view_size", view_size},
    {"view_indexing", view_indexing},
    {"view_iteration", view_iteration},
    {"view_arithmetic", view_arithmetic},
    {"stepped_slice", stepped_slice},
    {"stepped_slice_remainder", stepped_slice_remainder},
    {"inner_axis_slice", inner_axis_slice},
    {"multi_axis_slice", multi_axis_slice},
    {"nested_slice", nested_slice},
    {"out_of_bounds_index", out_of_bounds_index},
    {"zero_slice_step", zero_slice_step},
    {"empty_tensor", empty_tensor},
    {"scalar_tensor", scalar_tensor},
    {"incompatible_shapes", incompatible_shapes},
};

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: tensor_tests <test_case>\n";
        return 1;
    }

    for (const auto& test : tests) {
        if (test.name == argv[1]) {
            try {
                test.run();
                return 0;
            } catch (const std::exception& error) {
                std::cerr << test.name << ": " << error.what() << '\n';
                return 1;
            }
        }
    }

    std::cerr << "Unknown test case: " << argv[1] << '\n';
    return 1;
}
