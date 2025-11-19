from collections import InlineArray
from layout import Layout, LayoutTensor


alias rows = 3
alias cols = 3


fn matrix_row_vector_multiply(
    matrix: LayoutTensor[mut=True, DType.float64, Layout.row_major(rows, cols)],
    x: LayoutTensor[mut=True, DType.float64, Layout.row_major(rows, 1)],
    row: Int,
    size: Int,
) -> Float64:
    var acc: Float64 = 0.0
    for i in range(size):
        # tensor[x, y] returns a SIMD vector; [0] extracts the scalar
        acc += matrix[row, i][0] * x[i, 0][0]
    return acc


fn gauss_seidel(
    matrix: LayoutTensor[mut=True, DType.float64, Layout.row_major(rows, cols)],
    mut x: LayoutTensor[mut=True, DType.float64, Layout.row_major(rows, 1)],
    b: LayoutTensor[mut=True, DType.float64, Layout.row_major(rows, 1)],
    tolerance: Float64,
    size: Int,
    max_iterations: Int = 100,
):
    var err: Float64 = 1e308
    var iteration: Int = 0
    print("inside function")
    while err > tolerance and iteration < max_iterations:
        err = 0.0
        for i in range(size):
            var row_value = matrix_row_vector_multiply(matrix, x, i, size)
            var delta = ((
                b[i, 0][0]
                - row_value
                + matrix[i, i][0] * x[i, 0][0]
            ) / matrix[i, i][0]) - x[i, 0][0]
            err += delta * delta
            x[i, 0] = x[i, 0][0] + delta
        iteration += 1
        print("err", err)


fn main():
    # Simple 3x3 linear system Ax = b to demonstrate Gauss-Seidel, using
    # row-major LayoutTensors for matrix and vectors.
    alias layout_mat = Layout.row_major(rows, cols)
    alias layout_vec = Layout.row_major(rows, 1)

    var storage_A = InlineArray[Float64, rows * cols](uninitialized=True)
    var storage_b = InlineArray[Float64, rows](uninitialized=True)
    var storage_x = InlineArray[Float64, rows](uninitialized=True)

    var A = LayoutTensor[mut=True, DType.float64, layout_mat](storage_A)
    var b = LayoutTensor[mut=True, DType.float64, layout_vec](storage_b)
    var x = LayoutTensor[mut=True, DType.float64, layout_vec](storage_x)

    # Fill A with a diagonally dominant system.
    A[0, 0] = SIMD[DType.float64, 1](4.0)
    A[0, 1] = SIMD[DType.float64, 1](1.0)
    A[0, 2] = SIMD[DType.float64, 1](2.0)

    A[1, 0] = SIMD[DType.float64, 1](1.0)
    A[1, 1] = SIMD[DType.float64, 1](3.0)
    A[1, 2] = SIMD[DType.float64, 1](1.0)

    A[2, 0] = SIMD[DType.float64, 1](2.0)
    A[2, 1] = SIMD[DType.float64, 1](1.0)
    A[2, 2] = SIMD[DType.float64, 1](5.0)

    # b vector.
    b[0, 0] = SIMD[DType.float64, 1](4.0)
    b[1, 0] = SIMD[DType.float64, 1](2.0)
    b[2, 0] = SIMD[DType.float64, 1](4.0)

    # Initial guess x0 = 0.
    for i in range(rows):
        x[i, 0] = SIMD[DType.float64, 1](0.0)

    var size: Int = rows
    gauss_seidel(A, x, b, 1e-6, size, 100)
    for i in range(size):
        print("x[", i, "] =", x[i, 0][0])
