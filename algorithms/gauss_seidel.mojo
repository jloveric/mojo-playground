from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand
from math.limit import inf

fn matrix_row_vector_multiply[T:DType](
    matrix: Tensor[T], x: Tensor[T], row: Int, size: Int
)->Float64:
    var acc: T = 0.0
    for i in range(size):
        acc+=matrix[row,i]*x[i]
    
    return acc


"""
gauss_seidel operator. Trying to make this templated, but due to
some current limitations in mojo I need to define my scalars with
the particular type.
"""
fn gauss_seidel[T: DType](
    matrix: Tensor[T],
    inout x: Tensor[T],
    b: Tensor[T],
    tolerance: DType.float64,
    size: Int,
    max_iterations: Int=100
):
    var delta : Float64
    var err =inf[DType.float64]
    var iteration : Int = 0
    print('inside function')
    while err > tolerance and iteration<max_iterations:
        err = 0.0
        for i in range(size):
            delta = ((b[i] - matrix_row_vector_multiply[T](matrix,x,i,size) + matrix[i, i] * x[i]) / matrix[i, i])-x[i]
            err+=delta*delta
            x[i]+=delta
        iteration+=1
        print('err', err)

fn main() :
    let size = 100
    let m = rand[DType.float64](size, size)
    var x = rand[DType.float64](size)
    let b = rand[DType.float64](size)

    gauss_seidel[DType.float64](m, x, b, 1e-3, size, 100)
"""

fn main() :
    pass