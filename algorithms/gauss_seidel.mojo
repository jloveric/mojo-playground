from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand


fn matrix_row_vector_multiply(
    matrix: Tensor[DType.float64], x: Tensor[DType.float64], row: Int, size: Int
)->Float64:
    var acc: Float64 = 0.0
    for i in range(size):
        acc+=matrix[row,i]*x[i]
    
    return acc


fn gauss_seidel[T: DType](
    matrix: Tensor[T],
    inout x: Tensor[T],
    b: Tensor[T],
    tolerance: T,
    size: Int,
    max_iterations: Int=100
):
    var delta : T = 10*tolerance # Need max float in mojo
    var err : T=10*tolerance
    var iteration : Int = 0
    print('inside function')
    while err > tolerance and iteration<max_iterations:
        err = 0.0
        for i in range(size):
            delta = ((b[i] - matrix_row_vector_multiply(matrix,x,i,size) + matrix[i, i] * x[i]) / matrix[i, i])-x[i]
            err+=delta*delta
            x[i]+=delta
        iteration+=1
        print('err', err)

fn main() :
    let size = 100
    let m = rand[DType.float64](size, size)
    var x = rand[DType.float64](size)
    let b = rand[DType.float64](size)

    gauss_seidel[Float64](m, x, b, 1e-3, size, 100)