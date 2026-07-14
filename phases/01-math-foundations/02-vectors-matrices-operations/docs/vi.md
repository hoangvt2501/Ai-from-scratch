# Vectors, Ma trận & Hoạt động

> Mỗi mạng nơ-ron chỉ là phép nhân ma trận với các bước bổ sung.

**Loại:** Xây dựng
**Ngôn ngữ:** Python, Julia
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 01 (Trực giác đại số tuyến tính)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xây dựng class Ma trận với các phép toán theo phần tử, phép nhân ma trận, chuyển vị, định thức và nghịch đảo
- Phân biệt phép nhân theo nguyên tố với phép nhân ma trận và giải thích khi mỗi phép nhân áp dụng
- Triển khai một lớp mạng nơ-ron dày đặc duy nhất (`relu(W @ x + b)`) chỉ sử dụng Ma trận class từ đầu
- Giải thích các quy tắc phát sóng và cách hoạt động bias phép cộng trong mạng nơ-ron frameworks

## Vấn đề

Bạn muốn xây dựng một mạng nơ-ron. Bạn đọc mã và thấy điều này:

```
output = activation(weights @ input + bias)
```

`@` đó là phép nhân ma trận. Các `weights` là một ma trận. `input` là một vector. Nếu bạn không biết những thao tác đó làm gì, dòng này là phép thuật. Nếu bạn biết, nó là toàn bộ forward pass của một lớp trong ba hoạt động.

Mỗi hình ảnh model processes của bạn là một ma trận các giá trị pixel. Mỗi từ embedding là một vector. Mỗi lớp của mỗi mạng nơ-ron là một phép biến đổi ma trận. Bạn không thể xây dựng các hệ thống AI mà không thông thạo các hoạt động ma trận giống như cách bạn không thể viết mã nếu không hiểu các biến.

Bài học này xây dựng sự trôi chảy đó từ đầu.

## Khái niệm

### Vectors: danh sách các số được sắp xếp theo thứ tự

vector là danh sách các số có hướng và độ lớn. Trong AI, vectors đại diện cho các điểm dữ liệu, features hoặc parameters.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Một vector `[3, 4]` 2D trỏ đến tọa độ (3, 4) trên một mặt phẳng. Chiều dài (độ lớn) của nó là 5 (tam giác 3-4-5).

### Ma trận: lưới số

Ma trận là lưới 2D. Hàng và cột. Ma trận m x n có m hàng và n cột.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Trong mạng nơ-ron, ma trận trọng số biến đổi vectors đầu vào thành vectors đầu ra. Một lớp có 784 đầu vào và 128 đầu ra sử dụng ma trận trọng lượng 128x784.

### Tại sao hình dạng lại quan trọng

Phép nhân ma trận có một quy tắc nghiêm ngặt: `(m x n) @ (n x p) = (m x p)`. Kích thước bên trong phải phù hợp.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Nếu bạn gặp lỗi hình dạng không khớp trong PyTorch, đây là lý do tại sao.

### Bản đồ hoạt động

| hoạt động | Chức năng | Sử dụng mạng nơ-ron |
|-----------|-------------|-------------------|
| Bổ sung | Kết hợp phần tử khôn ngoan | Thêm bias vào đầu ra |
| Nhân vô hướng | Mở rộng mọi yếu tố | Learning rate * gradients |
| Ma trận nhân | Biến đổi vectors | Lớp forward pass |
| Chuyển vị | Lật hàng và cột | Backpropagation |
| Yếu tố quyết định | Tóm tắt số đơn | Kiểm tra khả năng đảo ngược |
| Nghịch đảo | Hoàn tác chuyển đổi | Giải quyết các hệ thống tuyến tính |
| Danh tính | Ma trận không làm gì | Khởi tạo, kết nối còn lại |

### Phép nhân theo nguyên tố so với ma trận

Sự khác biệt này liên tục khiến người mới bắt đầu vấp ngã.

Yếu tố khôn ngoan: nhân các vị trí phù hợp. Cả hai ma trận phải có cùng hình dạng.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Phép nhân ma trận: tích chấm của hàng và cột. Kích thước bên trong phải khớp.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Các hoạt động khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Phát thanh truyền hình

Khi bạn thêm bias vector vào ma trận đầu ra, các hình dạng không khớp. Phát sóng kéo dài mảng nhỏ hơn để phù hợp.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều tự động làm điều này. Hiểu nó ngăn ngừa nhầm lẫn khi các hình dạng có vẻ sai nhưng mã chạy.

```figure
vector-projection
```

## Tự xây dựng

### Bước 1: Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Ma trận class với các hoạt động cốt lõi

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Xem nó hoạt động

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Kết nối với mạng nơ-ron

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây là một lớp dày đặc duy nhất: `output = relu(W @ x + b)`. Mỗi lớp dày đặc trong mọi mạng nơ-ron đều thực hiện chính xác điều này.

## Ứng dụng

NumPy làm mọi thứ ở trên với ít dòng hơn và nhanh hơn.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Tổng đài `@` trong Python gọi `__matmul__`. NumPy triển khai nó với các quy trình BLAS được tối ưu hóa được viết bằng C và Fortran. Cùng một phép toán, nhanh hơn 100 lần.

Phát sóng ở NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động phát bias 1D trên cả hai hàng. Đây là cách phép cộng bias hoạt động trong mọi framework mạng nơ-ron.

## Sản phẩm bàn giao

Bài học này tạo ra một prompt để dạy các phép toán ma trận thông qua trực giác hình học. Xem `outputs/prompt-matrix-operations.md`.

Ma trận class xây dựng ở đây là nền tảng cho mạng nơ-ron mini framework chúng tôi xây dựng trong Giai đoạn 3, Bài 10.

## Bài tập

1. **Xác minh nghịch đảo.** Nhân `A @ A.inverse_2x2()` và xác nhận bạn nhận được ma trận nhận dạng. Hãy thử nó với ba ma trận 2x2 khác nhau. Điều gì xảy ra khi quyết định bằng không?

2. **Triển khai nghịch đảo 3x3.** Mở rộng class Ma trận để tính nghịch đảo cho ma trận 3x3 bằng phương pháp điều trị. Kiểm tra nó với `np.linalg.inv` của NumPy.

3. **Xây dựng mạng hai lớp.** Chỉ sử dụng class Ma trận của bạn (không có NumPy), tạo mạng nơ-ron hai lớp: đầu vào (3) -> ẩn (4) -> đầu ra (2). Khởi tạo trọng số ngẫu nhiên, chạy forward pass và xác minh tất cả các hình dạng là chính xác.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Vector | "Một mũi tên" | Một danh sách các số có thứ tự. Trong AI: một điểm trong không gian high-dimensional. |
| Ma trận | "Một bảng số" | Một phép biến đổi tuyến tính. Nó ánh xạ vectors từ không gian này sang không gian khác. |
| Ma trận nhân | "Chỉ cần nhân các con số" | Chấm các sản phẩm giữa mọi hàng của ma trận đầu tiên và mọi cột của ma trận thứ hai. Vấn đề trật tự. |
| Chuyển vị | "Lật nó" | Hoán đổi hàng và cột. Biến ma trận m x n thành n x m. Quan trọng trong backpropagation. |
| Yếu tố quyết định | "Một số từ ma trận" | Đo tỷ lệ ma trận bao nhiêu diện tích (2D) hoặc volume (3D). Không có nghĩa là sự biến đổi nghiền nát một chiều. |
| Nghịch đảo | "Hoàn tác ma trận" | Ma trận đảo ngược quá trình biến đổi. Chỉ tồn tại khi định thức không bằng không. |
| Ma trận nhận dạng | "Ma trận nhàm chán" | Ma trận tương đương với nhân với 1. Được sử dụng trong các kết nối dư (ResNets). |
| Phát thanh truyền hình | "Sửa hình dạng kỳ diệu" | Kéo dài một mảng nhỏ hơn để khớp với một mảng lớn hơn bằng cách lặp lại dọc theo các kích thước bị thiếu. |
| Yếu tố khôn ngoan | "Phép nhân đều đặn" | Nhân các vị trí phù hợp. Cả hai mảng phải có cùng hình dạng (hoặc có thể phát được). |

## Đọc thêm

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - trực giác trực quan cho mọi hoạt động được đề cập ở đây
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - các quy tắc chính xác NumPy tuân theo
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - tài liệu tham khảo ngắn gọn cho đại số tuyến tính cụ thể ML
