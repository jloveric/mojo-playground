from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand

"""
let height = 256
let width = 256
let channels = 3

# Create the tensor of dimensions height, width, channels
# and fill with random values.
let image = rand[DType.float32](height, width, channels)

# Declare the grayscale image.
let spec = TensorSpec(DType.float32, height, width)
var gray_scale_image = Tensor[DType.float32](spec)

# Perform the RGB to grayscale transform.
for y in range(height):
    for x in range(width):
        let r = image[y, x, 0]
        let g = image[y, x, 1]
        let b = image[y, x, 2]
        gray_scale_image[Index(y, x)] = 0.299 * r + 0.587 * g + 0.114 * b

print(gray_scale_image.shape().__str__())
"""


fn matrix_row_vector_multiply(
    matrix: Tensor[DType.float64], x: Tensor[DType.float64], row: Int, size: Int
)->Float64:
    var acc: Float64 = 0.0
    for i in range(size):
        acc+=matrix[row,i]*x[i]
    
    return acc


fn gauss_seidel(
    matrix: Tensor[DType.float64],
    x: Tensor[DType.float64],
    b: Tensor[DType.float64],
    tolerance: Float64,
    size: Int,  
):
    var delta : Float64
    var err : Float64=0.0
    for i in range(size):
        delta = ((b[i] - matrix_row_vector_multiply(matrix,x,i,size) + matrix[i, i] * x[i]) / matrix[i, i])-x[i]
        err+=delta*delta
        x[i]+=delta
