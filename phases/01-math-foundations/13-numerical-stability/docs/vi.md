# Độ ổn định số

> Dấu phẩy động là một trừu tượng bị rò rỉ. Nó sẽ cắn bạn trong training, và bạn sẽ không thấy nó đến.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 01-04
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Thực hiện softmax ổn định về số và log-sum-exp bằng thủ thuật trừ tối đa
- Xác định tràn, tràn và hủy bỏ thảm khốc trong tính toán dấu phẩy động
- Xác minh gradients phân tích so với gradients số bằng cách sử dụng chênh lệch hữu hạn căn giữa
- Giải thích lý do tại sao bfloat16 được ưa chuộng hơn float16 cho training và cách chia tỷ lệ loss ngăn chặn tràn gradient

## Vấn đề

model của bạn luyện tập trong ba giờ, sau đó loss trở thành NaN. Bạn thêm một bảng sao kê in. Các logits vẫn ổn ở bước 9,000. Ở bước 9.001, họ `inf`. Đến bước 9.002, mọi gradient đều `nan` và training đã chết.

Hoặc: model của bạn huấn luyện đến khi hoàn thành nhưng accuracy kém hơn 2% so với tuyên bố trên giấy. Bạn kiểm tra mọi thứ. Kiến trúc phù hợp. Hyperparameters trận đấu. Dữ liệu khớp. Vấn đề là giấy đã sử dụng float32 và bạn đã sử dụng float16 mà không có tỷ lệ phù hợp. Ba mươi hai lỗi làm tròn tích lũy lặng lẽ ăn mòn accuracy của bạn.

Hoặc: bạn thực hiện loss entropy chéo từ đầu. Nó hoạt động trên logits nhỏ. Khi logits vượt quá 100, nó sẽ trả về `inf`. Các softmax bị tràn vì `exp(100)` lớn hơn float32 có thể đại diện. Mỗi ML framework xử lý điều này bằng một thủ thuật hai dòng. Bạn không biết mánh khóe tồn tại.

Sự ổn định về số không phải là một mối quan tâm về mặt lý thuyết. Đó là sự khác biệt giữa một cuộc chạy training thành công và một cuộc chạy âm thầm thất bại. Mọi lỗi ML nghiêm trọng mà bạn sẽ gỡ lỗi cuối cùng đều đi xuống dấu phẩy động.

## Khái niệm

### IEEE 754: Cách máy tính lưu trữ số thực

Máy tính lưu trữ số thực dưới dạng giá trị dấu phẩy động theo tiêu chuẩn IEEE 754. Một phao có ba phần: một bit dấu, một số mũ và một bọ ngựa (ký hiệu).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Mantissa xác định precision (có bao nhiêu chữ số có nghĩa). Số mũ xác định phạm vi (một số có thể lớn hay nhỏ).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 cung cấp cho bạn khoảng 7 chữ số thập phân của precision. Điều đó có nghĩa là nó có thể phân biệt 1.0000001 và 1.0000002, nhưng không phải 1.00000001 và 1.00000002. Sau 7 chữ số, mọi thứ đều là nhiễu làm tròn.

float16 cung cấp cho bạn khoảng 3 chữ số. Con số lớn nhất mà nó có thể đại diện là 65.504. Điều đó nhỏ một cách đáng lo ngại đối với ML nơi logits, gradients và kích hoạt thường xuyên vượt quá mức này.

bfloat16 là câu trả lời của Google cho vấn đề phạm vi của float16. Nó có cùng số mũ 8 bit như float32 (cùng phạm vi, lên đến 3.4e38) nhưng chỉ có 7 bit mantissa (ít precision hơn float16). Đối với training mạng nơ-ron, phạm vi quan trọng hơn precision, vì vậy bfloat16 thường chiến thắng.

### Tại sao lại là 0,1 + 0,2 != 0,3

Số 0.1 không thể được biểu diễn chính xác trong dấu phẩy động nhị phân. Trong cơ số 2, nó là một phân số lặp lại:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 cắt bớt điều này xuống còn 23 bit mantissa. Giá trị được lưu trữ là khoảng 0,100000001490116. Tương tự, 0.2 được lưu trữ dưới dạng xấp xỉ 0.200000002980232. Tổng của chúng là 0,300000004470348, không phải 0,3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Điều này quan trọng đối với ML vì:

1. Loss so sánh như `if loss < threshold` có thể đưa ra câu trả lời sai
2. Tích lũy nhiều giá trị nhỏ (gradient cập nhật qua hàng nghìn bước) lệch so với tổng thực
3. Tổng kiểm tra và kiểm tra khả năng tái tạo không thành công nếu bạn so sánh float với `==`

Cách khắc phục: không bao giờ so sánh phao với `==`. Sử dụng `abs(a - b) < epsilon` hoặc `math.isclose()`.

### Hủy bỏ thảm họa

Khi bạn trừ đi hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa sẽ hủy bỏ và bạn sẽ bị nhiễu làm tròn được nâng lên các chữ số đứng đầu.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Đó là sai số tương đối 19% từ một phép trừ duy nhất. Trong ML, điều này xảy ra bất cứ khi nào bạn:

- Tính toán variance dữ liệu có giá trị trung bình lớn: `E[x^2] - E[x]^2` khi E[x] lớn
- Trừ đi xác suất log gần như bằng nhau
- Tính toán gradients chênh lệch hữu hạn với epsilon quá nhỏ

Cách khắc phục: sắp xếp lại các công thức để tránh trừ đi các số lớn, gần bằng nhau. Đối với variance, hãy sử dụng thuật toán Welford hoặc căn giữa dữ liệu trước. Đối với xác suất nhật ký, hãy làm việc trong không gian nhật ký xuyên suốt.

### Tràn và tràn

Tràn xảy ra khi một kết quả quá lớn để thể hiện. Dòng chảy dưới xảy ra khi nó quá nhỏ (gần bằng không hơn số dương có thể biểu diễn nhỏ nhất).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Hàm `exp()` là nguồn chính của tràn trong ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Hàm `log()` đánh theo hướng khác:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

Trong ML, `exp()` xuất hiện trong các tính toán softmax, sigmoid và xác suất. `log()` xuất hiện trong entropy chéo, log-likelihoods và phân kỳ KL. Sự kết hợp `log(exp(x))` là một bãi mìn không có thủ thuật phù hợp.

### Thủ thuật Log-Sum-Exp

Tính toán `log(sum(exp(x_i)))` trực tiếp rất nguy hiểm về mặt số. Nếu bất kỳ `x_i` nào lớn, `exp(x_i)` bị tràn. Nếu tất cả các `x_i` đều rất âm, mọi `exp(x_i)` đều chảy xuống bằng không và `log(0)` là `-inf`.

Bí quyết: trừ đi giá trị lớn nhất trước khi cấp mũ.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Tại sao điều này hoạt động: sau khi trừ đi `max(x)`, số mũ lớn nhất là `exp(0) = 1`. Không thể tràn. Ít nhất một số hạng trong tổng là 1, vì vậy tổng ít nhất là 1 và `log(1) = 0`. Không thể chảy xuống `-inf`.

Bằng chứng:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Đặt `c = max(x)` và tràn được loại bỏ.

Thủ thuật này xuất hiện ở khắp mọi nơi trong ML:
- Softmax chuẩn hóa
- Tính toán loss entropy chéo
- Tổng xác suất log theo trình tự models
- Hỗn hợp Gaussian
- Biến đổi inference

### Tại sao Softmax cần thủ thuật trừ tối đa

Softmax chuyển đổi logits thành xác suất:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Nếu không có mẹo, logits của [100, 101, 102] gây tràn:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Với thủ thuật, trừ max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Xác suất giống hệt nhau. Tính toán an toàn. Đây không phải là một tối ưu hóa. Đó là một yêu cầu về tính đúng đắn.

### NaN và Inf: Phát hiện và Phòng ngừa

`nan` (Không phải Số) và `inf` (vô cực) lan truyền virus thông qua tính toán. Một `nan` trong bản cập nhật gradient làm cho trọng lượng `nan`, điều này làm cho mọi đầu ra tiếp theo `nan`. Training đã chết trong vòng một bước.

Cách `inf` xuất hiện:
- `exp()` của một số dương lớn
- Chia cho không: `1.0 / 0.0`
- `float32` tràn trong tích tụ

Cách `nan` xuất hiện:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` số âm
- `log()` số âm
- Bất kỳ số học nào liên quan đến một `nan` hiện có

Phát hiện:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Chiến lược phòng ngừa:

1. Kẹp đầu vào để `exp()`: `exp(clamp(x, -80, 80))`
2. Thêm epsilon vào mẫu số: `x / (y + 1e-8)`
3. Thêm epsilon vào bên trong `log()`: `log(x + 1e-8)`
4. Sử dụng triển khai ổn định (log-sum-exp, softmax ổn định)
5. Gradient cắt để tránh nổ trọng lượng
6. Kiểm tra `nan`/`inf` sau mỗi forward pass trong quá trình gỡ lỗi

### Kiểm tra Gradient số

Các gradients phân tích (từ backpropagation) có thể có lỗi. Kiểm tra gradient số xác minh chúng bằng cách tính toán gradients có chênh lệch hữu hạn.

Công thức chênh lệch giữa:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Đây là O (h ^ 2) chính xác, tốt hơn nhiều so với `(f(x+h) - f(x)) / h` hiệu thuận chỉ là O (h).

Chọn h: quá lớn và xấp xỉ sai. Việc hủy bỏ quá nhỏ và thảm khốc sẽ phá hủy câu trả lời. `h = 1e-5` `1e-7` là điển hình.

Kiểm tra: tính toán sự khác biệt tương đối giữa gradients phân tích và số.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Quy tắc ngón tay cái:
- relative_error < 1e-7: hoàn hảo, gradient đúng
- relative_error < 1e-5: chấp nhận được, có thể đúng
- relative_error > 1e-3: Có điều gì đó không ổn
- relative_error > 1: gradient hoàn toàn sai

Luôn kiểm tra gradients khi triển khai một lớp hoặc chức năng loss mới. PyTorch cung cấp `torch.autograd.gradcheck()` cho việc này.

### Mixed Precision Training

Các GPUs hiện đại có phần cứng chuyên dụng (lõi Tensor) tính toán phép nhân ma trận float16 nhanh hơn 2-8 lần so với float32. Mixed precision training khai thác điều này:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Vấn đề với float16 thuần túy training: gradients thường rất nhỏ (1e-8 hoặc nhỏ hơn). Float16 làm giảm bất kỳ thứ gì dưới ~6e-8 về không. model của bạn ngừng học vì tất cả các bản cập nhật gradient đều bằng không.

Cách khắc phục là loss mở rộng quy mô:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Tỷ lệ loss động tự động điều chỉnh hệ số tỷ lệ. Bắt đầu với một giá trị lớn (65536). Nếu gradients tràn đến `inf`, hãy giảm một nửa. Nếu N bước vượt qua mà không bị tràn, hãy nhân đôi nó.

### bfloat16 vs float16: Tại sao bfloat16 thắng Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 có nhiều precision hơn (10 bit mantissa so với 7) nhưng phạm vi hạn chế (tối đa ~65,504). bfloat16 có ít precision hơn nhưng cùng phạm vi với float32 (tối đa ~3.4e38).

Đối với training mạng nơ-ron:

- Kích hoạt và logits thường xuyên vượt quá 65.504 trong training tăng đột biến. nổi16 tràn; bfloat16 xử lý nó.
- Tỷ lệ Loss là bắt buộc với float16 nhưng thường không cần thiết với bfloat16 vì phạm vi của nó bao gồm phổ cường độ gradient.
- bfloat16 là một sự cắt bớt đơn giản của float32: thả 16 bit dưới cùng của mantissa. Chuyển đổi là tầm thường và không mất dữ liệu trong số mũ.

float16 được ưu tiên cho các inference trong đó các giá trị được giới hạn và precision quan trọng hơn. bfloat16 được ưa chuộng cho training nơi phạm vi quan trọng hơn. Đây là lý do tại sao NVIDIA GPUs TPUs và hiện đại (A100, H100) có hỗ trợ bfloat16 gốc.

### Gradient Cắt

Sự bùng nổ gradients xảy ra khi gradients phát triển theo cấp số nhân thông qua nhiều lớp (phổ biến trong RNN, mạng sâu và transformers). Một gradient lớn duy nhất có thể làm hỏng tất cả các trọng lượng trong một bước.

Hai loại cắt:

**Kẹp theo giá trị: **kẹp từng phần tử gradient một cách độc lập.

```
grad = clamp(grad, -max_val, max_val)
```

Đơn giản nhưng có thể thay đổi hướng của gradient vector.

**Cắt theo định mức: **chia tỷ lệ toàn bộ gradient vector để định mức của nó không vượt quá ngưỡng.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Giữ nguyên hướng của gradient. Đây là những gì `torch.nn.utils.clip_grad_norm_()` làm. Đó là sự lựa chọn tiêu chuẩn.

Các giá trị điển hình: `max_norm=1.0` cho transformers, `max_norm=0.5` cho RL `max_norm=5.0` cho các mạng đơn giản hơn.

Gradient cắt không phải là một bản hack. Đó là một cơ chế an toàn. Nếu không có nó, một ngoại lệ duy nhất batch có thể tạo ra một gradient đủ lớn để làm hỏng nhiều tuần training.

### Các lớp chuẩn hóa làm chất ổn định số

Batch chuẩn hóa, chuẩn hóa lớp và chuẩn hóa RMS thường được trình bày dưới dạng bộ chính quy giúp training hội tụ. Chúng cũng là chất ổn định số.

Nếu không chuẩn hóa, các hoạt tính có thể phát triển hoặc thu nhỏ theo cấp số nhân thông qua các lớp:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Chuẩn hóa tập trung lại và điều chỉnh lại kích hoạt ở mọi lớp:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (thường là 1e-5) ngăn chặn sự chia bằng không khi tất cả các hoạt động giống hệt nhau. Các parameters `gamma` đã học và `beta` cho phép mạng khôi phục bất kỳ quy mô nào nó cần.

Điều này giữ cho các giá trị trong phạm vi an toàn về mặt số trên toàn mạng, ngăn chặn cả tràn trong forward pass và gradient bùng nổ trong backward pass.

### Lỗi số ML phổ biến

**Lỗi: Loss là NaN sau vài epochs.
Nguyên nhân: logits phát triển quá lớn, softmax tràn. Hoặc learning rate quá cao và trọng số phân kỳ.
Khắc phục: sử dụng softmax ổn định (trừ tối đa), giảm learning rate, thêm gradient cắt.

**Lỗi: Loss bị kẹt tại log(num_classes).**
Nguyên nhân: model đầu ra là xác suất gần như đồng đều. Thường có nghĩa là gradients đang biến mất hoặc model không học hỏi chút nào.
Khắc phục: kiểm tra xem nhãn dữ liệu có chính xác không, xác minh chức năng loss, kiểm tra ReLU đã chết.

**Lỗi: accuracy xác thực thấp hơn dự kiến 1-3%.**
Nguyên nhân: mixed precision mà không có tỷ lệ loss thích hợp. Gradient underflow âm thầm loại bỏ các bản cập nhật nhỏ.
Khắc phục: bật tỷ lệ loss động hoặc chuyển sang bfloat16.

**Lỗi: Gradient định mức là 0.0 cho một số lớp.
Nguyên nhân: các tế bào thần kinh ReLU chết (tất cả các đầu vào đều âm), hoặc float16 underflow.
Khắc phục: sử dụng LeakyReLU hoặc GELU, sử dụng tỷ lệ gradient, kiểm tra khởi tạo trọng lượng.

**Lỗi: Model hoạt động trên một GPU nhưng cho kết quả khác nhau trên một  khác.
Nguyên nhân: thứ tự tích lũy dấu phẩy động không xác định. GPU giảm song song tổng theo các thứ tự khác nhau trên các phần cứng khác nhau và phép cộng dấu phẩy động không liên kết.
Khắc phục: chấp nhận sự khác biệt nhỏ (1e-6), hoặc đặt `torch.use_deterministic_algorithms(True)` và chấp nhận hình phạt tốc độ.

**Lỗi: `exp()` trả về `inf` trong tính toán loss.**
Nguyên nhân: nguyên logits thô được chuyển sang `exp()` mà không có thủ thuật trừ tối đa.
Khắc phục: sử dụng `torch.nn.functional.log_softmax()` triển khai log-sum-exp nội bộ.

**Lỗi: Training phân kỳ sau khi chuyển từ float32 sang float16.
Nguyên nhân: float16 không thể đại diện cho gradient cường độ dưới 6e-8 hoặc kích hoạt trên 65.504.
Khắc phục: sử dụng mixed precision với tỷ lệ loss (AMP) hoặc sử dụng bfloat16 để thay thế.

```figure
logsumexp-stability
```

## Tự xây dựng

### Bước 1: Chứng minh giới hạn precision dấu phẩy động

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Bước 2: Triển khai softmax ngây thơ và ổn định

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Bước 3: Triển khai log-sum-exp ổn định

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Bước 4: Triển khai entropy chéo ổn định

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Bước 5: Kiểm tra Gradient

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Ứng dụng

### Mô phỏng Mixed precision

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient cắt

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Phát hiện NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Xem `code/numerical.py` để biết cách triển khai hoàn chỉnh với tất cả các trường hợp biên được minh họa.

## Sản phẩm bàn giao

Bài học này tạo ra:
- `code/numerical.py` với softmax ổn định, log-sum-exp, cross-entropy, kiểm tra gradient và mô phỏng mixed precision
- `outputs/prompt-numerical-debugger.md` để chẩn đoán các vấn đề về NaN/Inf và số trong training

Các triển khai ổn định này xuất hiện lại trong Giai đoạn 3 khi xây dựng vòng lặp training và trong Giai đoạn 4 khi triển khai các cơ chế attention.

## Bài tập

1. **Hủy bỏ thảm họa.** Tính variance của [1000000.0, 1000001.0, 1000002.0] bằng cách sử dụng công thức ngây thơ `E[x^2] - E[x]^2` trong float32. Sau đó, tính toán nó bằng thuật toán trực tuyến của Welford. So sánh các sai số với variance thực (0,6667).

2. **Precision săn lùng.** Tìm giá trị float32 dương nhỏ nhất `x` sao cho `1.0 + x == 1.0` tính bằng Python. Đây là epsilon máy. Xác minh rằng nó khớp với `numpy.finfo(numpy.float32).eps`.

3. **Các trường hợp biên Log-sum-exp.** Kiểm tra hàm `logsumexp_stable` của bạn với: (a) tất cả các giá trị bằng nhau, (b) một giá trị lớn hơn nhiều so với rest, (c) tất cả các giá trị rất âm (-1000). Xác minh rằng nó cho kết quả chính xác khi phiên bản ngây thơ không thành công.

4. **Gradient Kiểm tra lớp mạng nơ-ron.** Triển khai một `y = Wx + b` lớp tuyến tính duy nhất và backward pass phân tích của nó. Sử dụng `numerical_gradient` để xác minh tính đúng đắn cho ma trận trọng lượng 3x2.

5. **Loss thử nghiệm tỷ lệ.** Mô phỏng training với float16: tạo gradients ngẫu nhiên trong phạm vi [1e-9, 1e-3], chuyển đổi thành float16 và đo phân số nào trở thành không. Sau đó áp dụng tỷ lệ loss (nhân với 1024), chuyển đổi thành float16, thu nhỏ lại và đo lại phân số không.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| IEEE 754 | "Tiêu chuẩn phao" | Tiêu chuẩn quốc tế xác định định dạng dấu phẩy động nhị phân, quy tắc làm tròn và giá trị đặc biệt (inf, nan). Mọi CPU và GPU hiện đại đều thực hiện nó. |
| Máy epsilon | "Giới hạn precision" | Giá trị nhỏ nhất e sao cho 1.0 + e != 1.0 ở định dạng float nhất định. Đối với float32, nó là khoảng 1,19e-7. |
| Hủy bỏ thảm khốc | "Precision loss từ phép trừ" | Khi trừ các số dấu phẩy động gần bằng nhau, các chữ số có nghĩa sẽ hủy bỏ và nhiễu làm tròn chiếm ưu thế trong kết quả. |
| Tràn | "Số quá lớn" | Kết quả vượt quá giá trị biểu diễn tối đa có thể biểu diễn và trở thành inf. exp(89) tràn float32. |
| Dòng chảy ngầm | "Số lượng quá nhỏ" | Kết quả gần bằng 0 hơn số dương có thể biểu diễn nhỏ nhất và trở thành 0.0. EXP (-104) Dòng chảy dưới Phao 32. |
| Thủ thuật Log-sum-exp | "Trừ tối đa trước" | Tính toán nhật ký (sum (exp (x))) bằng cách bao gồm exp (max (x)) để ngăn tràn và tràn vào. Được sử dụng trong toán học softmax, entropy chéo và xác suất log. |
| softmax ổn định | "Softmax không nổ" | Trừ tối đa (logits) trước khi cấp mũ. Kết quả giống hệt nhau về mặt số, không thể tràn. |
| Kiểm tra Gradient | "Xác minh backprop của bạn" | So sánh gradients phân tích từ backpropagation với số gradients từ sự khác biệt hữu hạn để bắt lỗi triển khai. |
| Mixed precision | "Float16 tiến, float32 lùi" | Sử dụng phao precision thấp hơn cho các hoạt động quan trọng về tốc độ và phao precision cao hơn cho các hoạt động nhạy cảm về số. Tăng tốc điển hình là 2-3x. |
| Loss mở rộng quy mô | "Ngăn chặn dòng chảy gradient" | Nhân loss với một hằng số lớn trước backprop để gradients ở trong phạm vi đại diện của float16, sau đó chia cho cùng một hằng số trước khi cập nhật trọng số. |
| bfloat16 | "Dấu phẩy động não" | Định dạng 16 bit của Google với 8 bit số mũ (cùng phạm vi với float32) và 7 bit mantissa (ít precision hơn float16). Ưu tiên cho training. |
| Gradient cắt | "Giới hạn tiêu chuẩn gradient" | Mở rộng quy mô gradient vector sao cho định mức của nó không vượt quá ngưỡng. Ngăn gradients nổ làm hỏng trọng lượng. |
| NaN | "Không phải là một con số" | Giá trị float đặc biệt từ các hoạt động không xác định (0/0, inf-inf, sqrt(-1)). Truyền bá qua tất cả các số học tiếp theo. |
| Inf | "Vô cực" | Giá trị float đặc biệt từ tràn hoặc chia cho không. Có thể kết hợp để tạo ra NaN (inf - inf, inf * 0). |
| gradient số | "Dẫn xuất vũ phu" | Xấp xỉ đạo hàm bằng cách đánh giá f(x+h) và f(xh) và chia cho 2h. Chậm nhưng đáng tin cậy để xác minh. |

## Đọc thêm

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- tài liệu tham khảo dứt khoát, dày đặc nhưng đầy đủ
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- bài báo NVIDIA giới thiệu tỷ lệ loss cho float16 training
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- hướng dẫn thực tế để mixed precision trong PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- tại sao Google chọn định dạng này cho TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- thuật toán để giảm sai số làm tròn trong tổng dấu phẩy động
