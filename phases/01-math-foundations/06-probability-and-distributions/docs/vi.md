# Xác suất và phân phối

> Xác suất là ngôn ngữ AI sử dụng để thể hiện sự không chắc chắn.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 01-04
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Triển khai PMF và PDF từ đầu cho các phân phối Bernoulli, phân loại, Poisson, đồng nhất và chuẩn
- Tính toán giá trị kỳ vọng, variance và sử dụng Định lý giới hạn trung tâm để giải thích tại sao Gaussian chiếm ưu thế
- Xây dựng các hàm softmax và log-softmax bằng thủ thuật ổn định số (trừ logit tối đa)
- Tính toán loss entropy chéo từ logits và kết nối nó với log-likelihood âm

## Vấn đề

Một bộ phân loại xuất ra `[0.03, 0.91, 0.06]`. Một model ngôn ngữ chọn từ tiếp theo từ 50.000 ứng viên. Một model khuếch tán tạo ra hình ảnh bằng sampling từ các bản phân phối đã học. Tất cả những điều này đều là xác suất trong hành động.

Mỗi dự đoán mà một model đưa ra là một phân phối xác suất. Mỗi hàm loss đo lường khoảng cách phân phối dự đoán so với phân phối thực. Mỗi training bước điều chỉnh parameters để làm cho một phân phối trông giống bản phân phối khác. Nếu không có xác suất, bạn không thể đọc một bài báo ML, gỡ lỗi một model hoặc hiểu tại sao training loss của bạn là NaN.

## Khái niệm

### Sự kiện, không gian mẫu và xác suất

Không gian mẫu S là tập hợp tất cả các kết quả có thể xảy ra. Sự kiện là một tập hợp con của không gian mẫu. Xác suất ánh xạ các sự kiện với các số từ 0 đến 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Ba tiên đề xác định tất cả xác suất:
1. P (A) > = 0 cho bất kỳ sự kiện A nào
2. P(S) = 1 (điều gì đó luôn xảy ra)
3. P (A hoặc B) = P (A) + P (B) khi A và B không thể xảy ra cả hai

Mọi thứ khác (định lý Bayes, kỳ vọng, phân phối) đều tuân theo ba quy tắc này.

### Xác suất có điều kiện và tính độc lập

P (A |B) là xác suất của A cho rằng B đã xảy ra.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Hai sự kiện độc lập khi biết một sự kiện không cho bạn biết gì về sự kiện kia:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Tung đồng xu là độc lập. Rút thẻ mà không thay thế thì không.

### Hàm khối lượng xác suất so với hàm mật độ xác suất

Các biến ngẫu nhiên rời rạc có hàm khối lượng xác suất (PMF). Mỗi kết quả có một xác suất cụ thể mà bạn có thể đọc trực tiếp.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Các biến ngẫu nhiên liên tục có hàm mật độ xác suất (PDF). Mật độ tại một điểm duy nhất không phải là xác suất. Xác suất đến từ việc tích hợp mật độ trong một khoảng thời gian.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Sự khác biệt này quan trọng trong ML. Đầu ra phân loại là PMF (lựa chọn rời rạc). Không gian tiềm ẩn VAE sử dụng PDF (liên tục).

### Phân phối phổ biến

**Bernoulli: **một thử nghiệm, hai kết quả. Models phân loại nhị phân.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Phân loại: **một thử nghiệm, k kết quả. Models phân loại đa class (đầu ra softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Đồng phục: **tất cả các kết quả đều có khả năng như nhau. Được sử dụng để khởi tạo ngẫu nhiên.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Bình thường (Gaussian):** đường cong chuông. Được tham số hóa bằng giá trị trung bình (mu) và variance (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** đếm các sự kiện hiếm gặp trong một khoảng thời gian cố định. Models tỷ lệ sự kiện.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Giá trị và Variance kỳ vọng

Giá trị kỳ vọng là kết quả bình quân gia quyền.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Các biện pháp Variance trải rộng xung quanh giá trung bình.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

Trong ML, giá trị kỳ vọng xuất hiện dưới dạng hàm loss (loss trung bình trên phân phối dữ liệu). Variance cho bạn biết về sự ổn định model. variance cao trong gradients có nghĩa là training ồn ào.

### Phân phối chung và cận biên

Phân phối chung P(X, Y) mô tả hai biến ngẫu nhiên với nhau.

Ví dụ PMF chung (X = thời tiết, Y = ô):

|| Y = 0 (không có ô) | Y = 1 (ô) | P (X) cận biên |
|---|---|---|---|
| X = 0 (mặt trời) | 0.40 | 0.10 | P (X = 0) = 0.50 |
| X=1 (mưa) | 0.05 | 0.45 | P (X = 1) = 0.50 |
| **P (Y) cận biên **| P (Y = 0) = 0,45 | P (Y = 1) = 0,55 | 1.00 |

Phân phối cận biên tổng hợp các biến khác:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Tổng số hàng và cột trong bảng trên là các phần bên lề.

### Tại sao phân phối chuẩn hiển thị ở khắp mọi nơi

Định lý giới hạn trung tâm: tổng (hoặc trung bình) của nhiều biến ngẫu nhiên độc lập hội tụ thành phân phối chuẩn, bất kể phân phối gốc.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Đây là lý do tại sao:
- Sai số đo gần như bình thường (nhiều nguồn độc lập nhỏ)
- Khởi tạo trọng lượng trong mạng nơ-ron sử dụng phân phối chuẩn
- Gradient nhiễu trong SGD là xấp xỉ bình thường (tổng của nhiều gradients mẫu)
- Phân phối chuẩn là phân phối entropy tối đa cho một giá trị trung bình và variance nhất định

### Log Probabilities

Xác suất thô gây ra các vấn đề về số. Nhân nhiều xác suất nhỏ với nhau nhanh chóng chảy xuống bằng không.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities khắc phục điều này. Phép nhân trở thành phép cộng.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Quy tắc:
- log(a * b) = log(a) + log(b)
- log probabilities luôn là <= 0 (vì 0 < P <= 1)
- Tiêu cực nhiều hơn = ít có khả năng hơn
- loss entropy chéo là log probability âm của class chính xác

### Softmax như một phân phối xác suất

Mạng nơ-ron xuất ra điểm thô (logits). Softmax chuyển đổi chúng thành một phân phối xác suất hợp lệ.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Thủ thuật softmax: trừ đi mức logit tối đa trước khi cấp số mũ để tránh tràn.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax kết hợp softmax và log để ổn định số. PyTorch sử dụng điều này bên trong cho loss entropy chéo.

### Sampling

Sampling có nghĩa là rút các giá trị ngẫu nhiên từ một phân phối. Trong ML:
- Dropout ngẫu nhiên samples mà các tế bào thần kinh không
- Các mẫu tăng cường dữ liệu biến đổi ngẫu nhiên
- Ngôn ngữ models lấy mẫu token tiếp theo từ phân phối dự đoán
- Khuếch tán models nhiễu mẫu và khử nhiễu dần dần

Sampling từ các phân phối tùy ý yêu cầu các kỹ thuật như sampling biến đổi nghịch đảo, sampling loại bỏ hoặc thủ thuật tái tham số hóa (được sử dụng trong VAE).

```figure
gaussian-pdf
```

## Tự xây dựng

### Bước 1: Khái niệm cơ bản về xác suất

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Bước 2: PMF và PDF từ đầu

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Bước 3: Giá trị kỳ vọng và variance

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Bước 4: Sampling từ các bản phân phối

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Bước 5: Softmax và log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Bước 6: Trình diễn Định lý giới hạn trung tâm

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Bước 7: Trực quan hóa

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Triển khai đầy đủ với tất cả các hình ảnh trực quan đều `code/probability.py`.

## Ứng dụng

Với NumPy và SciPy, mọi thứ trên đều là một dòng:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Bạn đã xây dựng những thứ này từ đầu. Bây giờ bạn biết các cuộc gọi thư viện đang làm gì.

## Bài tập

1. Triển khai sampling biến đổi nghịch đảo cho phân phối hàm mũ. Xác minh bằng cách sampling 10.000 giá trị và so sánh biểu đồ với PDF thực.

2. Xây dựng một bảng phân phối chung cho hai viên xúc xắc đã nạp. Tính toán các phân phối cận biên và kiểm tra xem xúc xắc có độc lập hay không.

3. Tính toán loss entropy chéo cho bộ phân loại 5 class xuất ra logits `[2.0, 0.5, -1.0, 3.0, 0.1]` khi class chính xác là chỉ số 3. Sau đó, xác minh câu trả lời của bạn với `nn.CrossEntropyLoss` của PyTorch.

4. Viết một hàm lấy danh sách các log probabilities và trả về dãy có khả năng xảy ra nhất, tổng log probability và xác suất thô tương đương. Kiểm tra nó với một câu 50 từ trong đó mỗi từ có xác suất 0,01.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Không gian mẫu | "Tất cả các khả năng" | Tập S của mọi kết quả có thể có của một thử nghiệm |
| PMF | "Hàm xác suất" | Một hàm cho xác suất chính xác của mỗi kết quả rời rạc, tổng thành 1 |
| Bản PDF | "Đường cong xác suất" | Một hàm mật độ cho các biến liên tục. Tích hợp nó trong một khoảng thời gian để có xác suất |
| Xác suất có điều kiện | "Xác suất cho một cái gì đó" | P (A \ | B) = P (A và B) / P (B). Nền tảng của tư duy Bayes và định lý Bayes |
| Độc lập | "Chúng không ảnh hưởng lẫn nhau" | P (A và B) = P (A) * P (B). Biết một sự kiện không cho bạn biết gì về sự kiện kia |
| Giá trị kỳ vọng | "Trung bình" | Tổng trọng số xác suất của tất cả các kết quả. Hàm loss là một giá trị kỳ vọng |
| Variance | "Làm thế nào trải rộng" | Độ lệch bình phương dự kiến so với giá trị trung bình. variance cao = ước tính ồn ào, không ổn định |
| Phân phối chuẩn | "Đường cong chuông" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Xuất hiện ở khắp mọi nơi do CLT |
| Định lý giới hạn trung tâm | "Mức trung bình trở nên bình thường" | Giá trị trung bình của nhiều mẫu độc lập hội tụ về phân phối chuẩn bất kể nguồn |
| Phân phối chung | "Hai biến với nhau" | P(X, Y) mô tả xác suất của mọi kết hợp kết quả X và Y |
| Phân phối cận biên | "Tổng ra biến số khác" | P (X) = sum_y P (X, Y). Khôi phục phân bố của một biến từ khớp |
| Log probability | "Nhật ký xác suất" | nhật ký P(x). Biến sản phẩm thành tổng, ngăn chặn dòng chảy số trong các chuỗi dài |
| Softmax | "Biến điểm số thành xác suất" | softmax (z_i) = EXP (z_i) / SUM (EXP (z_j)). Ánh xạ logits có giá trị thực với phân phối xác suất hợp lệ |
| Entropy chéo | "Chức năng loss" | -sum(p_true * log(p_predicted)). Đo lường mức độ khác nhau của hai phân phối. Thấp hơn là tốt hơn |
| Logits | "Đầu ra model thô" | Điểm số không chuẩn hóa trước softmax. Được đặt tên theo chức năng hậu cần |
| Sampling | "Rút các giá trị ngẫu nhiên" | Tạo ra các giá trị theo phân phối xác suất. Cách models tạo đầu ra |

## Đọc thêm

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - bằng chứng trực quan về lý do tại sao mức trung bình trở nên bình thường
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - tài liệu tham khảo ngắn gọn bao gồm mọi thứ ở đây và hơn thế nữa
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - tại sao độ ổn định số lại quan trọng và làm thế nào để đạt được nó
