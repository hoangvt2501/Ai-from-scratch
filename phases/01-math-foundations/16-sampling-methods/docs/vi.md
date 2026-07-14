# Sampling Phương pháp

> Sampling là cách AI khám phá không gian của các khả năng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 06-07 (Xác suất, Định lý Bayes)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Thực hiện CDF nghịch đảo, từ chối và sampling tầm quan trọng từ đầu chỉ bằng cách sử dụng các số ngẫu nhiên đồng nhất
- Xây dựng temperature, top-k và top-p (hạt nhân) sampling để tạo model token ngôn ngữ
- Giải thích thủ thuật tái tham số hóa và lý do tại sao nó cho phép backpropagation thông qua sampling trong VAE
- Chạy Metropolis-Hastings MCMC để lấy mẫu từ phân phối mục tiêu không chuẩn hóa

## Vấn đề

Một ngôn ngữ model hoàn thành việc xử lý prompt của bạn và tạo ra vector 50.000 logits. Một cho mỗi token trong từ vựng của nó. Bây giờ nó phải chọn một. Làm sao?

Nếu nó luôn chọn token xác suất cao nhất, mọi phản hồi đều giống hệt nhau. Xác định. Nhàm chán. Nếu nó chọn đồng nhất một cách ngẫu nhiên, đầu ra là vô nghĩa. Câu trả lời nằm ở đâu đó giữa những thái cực này, và ở đâu đó bị kiểm soát bởi sampling.

Sampling không giới hạn ở việc tạo văn bản. Học tăng cường ước tính policy gradients theo quỹ đạo sampling. VAE học các biểu diễn tiềm ẩn bằng cách sampling từ các bản phân phối đã học và lan truyền ngược thông qua tính ngẫu nhiên. Khuếch tán models tạo ra hình ảnh bằng cách sampling nhiễu và khử nhiễu lặp đi lặp lại. Phương pháp Monte Carlo ước tính các tích phân không có nghiệm dạng đóng. Các thuật toán MCMC khám phá các phân phối high-dimensional posterior không thể liệt kê.

Mỗi hệ thống AI tổng quát là một hệ thống sampling. Chiến lược sampling xác định chất lượng, sự đa dạng và khả năng kiểm soát của đầu ra. Bài học này xây dựng mọi phương pháp sampling chính từ đầu, bắt đầu từ các số ngẫu nhiên đồng nhất và kết thúc bằng các kỹ thuật cung cấp năng lượng cho LLMs hiện đại và models tạo sinh.

## Khái niệm

### Tại sao Sampling lại quan trọng

Sampling xuất hiện trong bốn vai trò cơ bản trong AI và học máy:

**Generation.** models ngôn ngữ, models khuếch tán và GAN đều tạo ra đầu ra theo sampling. Thuật toán sampling trực tiếp kiểm soát sự sáng tạo, mạch lạc và đa dạng. Temperature, top-k và sampling hạt nhân là những núm vặn mà các kỹ sư xoay hàng ngày.

**Training.** Stochastic gradient descent samples mini-batches. Dropout các tế bào thần kinh lấy mẫu để vô hiệu hóa. Tăng cường dữ liệu lấy mẫu biến đổi ngẫu nhiên. Tầm quan trọng sampling cân lại các mẫu để giảm gradient variance trong học tăng cường (PPO, TRPO).

**Ước tính.** Nhiều đại lượng trong ML không có dung dịch dạng đóng. Sự loss dự kiến trên một phân phối dữ liệu, hàm phân vùng của một model dựa trên năng lượng, bằng chứng trong inference Bayes. Ước tính Monte Carlo xấp xỉ tất cả những điều này bằng cách tính trung bình trên các mẫu.

**Khám phá.** Các thuật toán MCMC khám phá các phân phối posterior trong inference Bayes. Các chiến lược tiến hóa lấy mẫu parameter nhiễu loạn. Thompson sampling cân bằng việc thăm dò và khai thác trong bọn cướp.

Thách thức cốt lõi: bạn chỉ có thể lấy mẫu trực tiếp từ các phân phối đơn giản (đồng nhất, bình thường). Đối với mọi thứ khác, bạn cần một phương pháp để chuyển đổi các mẫu đơn giản thành mẫu từ phân phối mục tiêu của bạn.

### Đồng phục ngẫu nhiên Sampling

Mọi phương pháp sampling đều bắt đầu từ đây. Một trình tạo số ngẫu nhiên đồng nhất tạo ra các giá trị trong [0, 1) trong đó mọi khoảng phụ có độ dài bằng nhau đều có xác suất bằng nhau.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Để lấy mẫu đồng nhất từ một tập hợp n mục rời rạc, hãy tạo u và trả về floor(n * U). Để lấy mẫu từ một phạm vi liên tục [a, b], hãy tính a + (b - a) * U.

Thông tin chi tiết quan trọng: một số ngẫu nhiên đồng nhất duy nhất chứa chính xác lượng ngẫu nhiên phù hợp để tạo ra một mẫu từ bất kỳ phân phối nào. Bí quyết là tìm ra sự biến đổi phù hợp.

### Phương pháp CDF nghịch đảo (Biến đổi nghịch đảo Sampling)

Hàm phân phối tích lũy (CDF) ánh xạ các giá trị với xác suất:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

CDF nghịch đảo ánh xạ xác suất trở lại các giá trị. Nếu U ~ Uniform(0, 1), thì X = F_inverse(U) tuân theo phân phối đích.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Ví dụ phân phối theo cấp số nhân:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Điều này hoạt động hoàn hảo khi bạn có thể viết ra F_inverse ở dạng đóng. Đối với phân phối chuẩn, không có CDF nghịch đảo dạng đóng, vì vậy chúng tôi sử dụng các phương pháp khác (Box-Muller, hoặc xấp xỉ số).

**Phiên bản rời rạc:** Đối với các phân phối rời rạc, hãy xây dựng CDF dưới dạng tổng tích lũy, tạo U và tìm chỉ mục đầu tiên có tổng tích lũy vượt quá U. Đây là cách `sample_categorical` hoạt động trong Bài 06.

### Từ chối Sampling

Khi bạn không thể đảo ngược CDF nhưng có thể đánh giá PDF đích đến một hằng số, sampling từ chối sẽ hoạt động.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

M ràng buộc càng chặt chẽ thì tỷ lệ chấp nhận càng cao. Ở kích thước thấp (1-3), sampling loại bỏ hoạt động tốt. Trong các khía cạnh cao, tỷ lệ chấp nhận giảm theo cấp số nhân vì hầu hết các đề xuất volume bị từ chối. Đây là lời nguyền của không gian để từ chối sampling.

**Ví dụ: sampling từ một chuẩn bị cắt bớt.** Sử dụng một đề xuất thống nhất trên phạm vi bị cắt bớt. Phong bì M là giá trị tối đa của PDF bình thường trong phạm vi đó.

**Ví dụ: sampling từ một hình bán nguyệt.** Đề xuất thống nhất trong hình chữ nhật có viền. Chấp nhận nếu điểm nằm trong hình bán nguyệt. Đây là cách Monte Carlo tính pi: tỷ lệ chấp nhận bằng tỷ lệ diện tích pi/4.

### Tầm quan trọng Sampling

Đôi khi bạn không cần mẫu từ phân phối mục tiêu p(x). Bạn cần ước tính một kỳ vọng theo p(x) và bạn có các mẫu từ một phân phối khác q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Điều này rất quan trọng trong học tăng cường. Trong PPO (Tối ưu hóa Policy gần), bạn thu thập quỹ đạo dưới một policy pi_old cũ nhưng muốn tối ưu hóa một policy pi_new mới. Trọng số quan trọng là pi_new(a|s) / pi_old(a|s). PPO kẹp các trọng lượng này để ngăn policy mới phân kỳ quá xa so với cái cũ.

Sự variance về tầm quan trọng của ước tính sampling phụ thuộc vào mức độ tương tự của q với p. Nếu q rất khác với p, một vài mẫu có trọng số rất lớn và chiếm ưu thế trong ước tính. Tầm quan trọng tự chuẩn hóa sampling chia cho tổng trọng số để giảm vấn đề này:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Ước tính Monte Carlo

Ước tính Monte Carlo xấp xỉ tích phân bằng cách tính trung bình các mẫu ngẫu nhiên. Quy luật số lớn đảm bảo sự hội tụ.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tỷ lệ lỗi không phụ thuộc vào kích thước. Đây là lý do tại sao các phương pháp Monte Carlo chiếm ưu thế trong các chiều cao, nơi không thể tích hợp dựa trên lưới.

**Ước tính pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Ước tính kỳ vọng:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Chuỗi Markov Monte Carlo (MCMC): Metropolis-Hastings

MCMC xây dựng một chuỗi Markov có phân phối tĩnh là phân phối mục tiêu p(x). Sau đủ bước, các mẫu từ chuỗi (xấp xỉ) là các mẫu từ p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Đối với các đề xuất đối xứng (q(x'|x) = q(x|x')), tỷ lệ đơn giản hóa thành p(x')/p(x). Đây là thuật toán Metropolis ban đầu.

**Tại sao nó hoạt động.** Quy tắc chấp nhận đảm bảo sự cân bằng chi tiết: xác suất ở x và di chuyển đến x' bằng xác suất ở x' và di chuyển đến x. Cân bằng chi tiết ngụ ý rằng p(x) là sự phân bố tĩnh của chuỗi.

**Cân nhắc thực tế:**
- Burn-in: loại bỏ các mẫu sớm trước khi chuỗi đạt đến trạng thái cân bằng
- Làm mỏng: giữ mọi mẫu thứ k để giảm tự tương quan
- Quy mô đề xuất: quá nhỏ và chuỗi di chuyển chậm (chấp nhận cao, thăm dò chậm); quá lớn và hầu hết các đề xuất đều bị từ chối (chấp nhận thấp, bị mắc kẹt tại chỗ)
- Tỷ lệ chấp nhận tối ưu cho một đề xuất Gaussian ở kích thước cao là khoảng 0,234

### Gibbs Sampling

Gibbs sampling là một trường hợp đặc biệt của MCMC cho các phân phối đa biến. Thay vì đề xuất một bước di chuyển trong tất cả các chiều cùng một lúc, nó cập nhật một biến tại một thời điểm từ phân phối có điều kiện của nó.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling yêu cầu bạn có thể lấy mẫu từ mỗi phân phối có điều kiện p(x_i | x_{-i}). Điều này rất đơn giản đối với nhiều models:
- Mạng Bayes: các điều kiện theo sau cấu trúc đồ thị
- Hỗn hợp Gaussian: điều kiện là Gaussian
- Ising models: mỗi vòng quay có điều kiện chỉ phụ thuộc vào hàng xóm của nó

Tỷ lệ chấp nhận luôn là 1 (mọi đề xuất đều được chấp nhận) vì sampling từ điều kiện chính xác sẽ tự động thỏa mãn số dư chi tiết.

**Hạn chế.** Khi các biến có mối tương quan cao, Gibbs sampling trộn chậm vì cập nhật một biến tại một thời điểm không thể thực hiện các chuyển động chéo lớn thông qua phân phối.

### Temperature Sampling (Được sử dụng trong LLMs)

Ngôn ngữ models đầu ra logits z_1, ..., z_V cho từng token trong từ vựng. Softmax chuyển đổi chúng thành xác suất. Temperature thay đổi quy mô logits trước khi softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Tại sao nó hoạt động.** Chia logits cho T < 1 khuếch đại sự khác biệt giữa các logits. Nếu z_1 = 2 và z_2 = 1, chia cho T = 0,5 cho z_1/T = 4 và z_2/T = 2, làm cho khoảng cách lớn hơn. Sau softmax, logit token cao nhất nhận được một phần lớn hơn nhiều.

**Trong thực tế:**
- T = 0.0: giải mã tham lam, tốt nhất cho Q&A thực tế
- T = 0,3-0,7: hơi sáng tạo, tốt cho việc tạo mã
- T = 0.7-1.0: cân bằng, tốt cho cuộc trò chuyện chung
- T = 1.0-1.5: viết sáng tạo, động não
- T > 1.5: ngày càng ngẫu nhiên, hiếm khi hữu ích

Temperature không thay đổi tokens nào có thể. Nó thay đổi khối lượng xác suất được phân bổ cho mỗi token.

### Top-k Sampling

Top-k sampling giới hạn bộ ứng cử viên ở k tokens có xác suất cao nhất, sau đó chuẩn hóa lại và lấy mẫu từ tập hợp hạn chế đó.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k ngăn model chọn những tokens cực kỳ khó có thể xảy ra (lỗi chính tả, vô nghĩa) tồn tại trong đuôi dài của phân phối từ vựng. Vấn đề: k được cố định bất kể ngữ cảnh. Khi model tin cậy (một token có xác suất 95%), k = 40 vẫn cho phép 39 lựa chọn thay thế. Khi model không chắc chắn (xác suất trải rộng trên 1000 tokens), k = 40 cắt bỏ các lựa chọn hợp lý.

### Top-p (nhân) Sampling

Top-p sampling tự động điều chỉnh kích thước bộ ứng viên. Thay vì giữ một số tokens cố định, nó giữ tập tokens nhỏ nhất có xác suất tích lũy vượt quá p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Khi model tự tin, nhân sampling giữ ít tokens (có thể 2-3). Khi model không chắc chắn, nó giữ lại nhiều (có thể là 200). Hành vi thích ứng này là lý do tại sao sampling nhân thường tạo ra văn bản tốt hơn top-k.

**Kết hợp phổ biến:**
- Temperature 0,7 + top-p 0,9: cài đặt mục đích chung tốt
- Temperature 0.0 (tham lam): tốt nhất cho các nhiệm vụ xác định
- Temperature 1.0 + top-k 50: Cài đặt giấy gốc của Fan et al. (2018)

Top-k và top-p có thể được kết hợp. Áp dụng top-k trước, sau đó top-p lên phần còn lại.

### Thủ thuật tái tham số hóa (được sử dụng trong VAE)

Bộ mã hóa tự động biến đổi (VAE) học bằng cách mã hóa đầu vào vào một phân phối trong không gian tiềm ẩn, sampling từ phân phối đó và giải mã mẫu trở lại. Vấn đề: bạn không thể lan truyền ngược thông qua một hoạt động sampling.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Thủ thuật tái tham số hóa tách tính ngẫu nhiên khỏi parameters:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Điều này hoạt động vì N(mu, sigma^2) có cùng phân phối với mu + sigma * N(0, 1). Thông tin chi tiết quan trọng: di chuyển tính ngẫu nhiên đến nguồn không có parameter (epsilon), sau đó biểu thị mẫu dưới dạng một phép biến đổi có thể vi phân của parameters.

**Trong vòng lặp training VAE:**
1. Encoder đầu ra MU và Log (Sigma ^ 2) cho mỗi đầu vào
2. Mẫu epsilon ~ N (0, 1)
3. Tính toán z = mu + sigma * epsilon
4. Giải mã z để xây dựng lại đầu vào
5. Lan truyền ngược qua các bước 4, 3, 2, 1 (có thể vì bước 3 có thể phân biệt)

Nếu không có thủ thuật tái tham số, VAE không thể được huấn luyện với backpropagation tiêu chuẩn. Cái nhìn sâu sắc duy nhất này đã làm cho VAE trở nên thực tế.

### Gumbel-Softmax (Sampling phân loại có thể phân biệt)

Thủ thuật tái tham số hóa hoạt động đối với các phân phối liên tục (Gaussian). Đối với các phân phối phân loại rời rạc, chúng ta cần một cách tiếp cận khác. Gumbel-Softmax cung cấp một xấp xỉ có thể phân biệt được với sampling phân loại.

**Thủ thuật Gumbel-Max (không thể phân biệt):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (xấp xỉ có thể vi phân):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax tạo ra sự thư giãn liên tục của một mẫu rời rạc. Đầu ra là xác suất vector (mềm-nóng) thay vì một nóng. Gradients chảy qua softmax. Trong quá trình forward pass training, bạn có thể sử dụng công cụ ước tính "thẳng": sử dụng argmax cứng cho forward pass nhưng Gumbel-Softmax gradients mềm cho backward pass.

**Ứng dụng:**
- Các biến tiềm ẩn rời rạc trong VAE
- Tìm kiếm kiến trúc thần kinh (chọn các hoạt động rời rạc)
- Cơ chế attention cứng
- Học tăng cường với các hành động rời rạc

### Phân tầng Sampling

sampling Monte Carlo tiêu chuẩn có thể để lại khoảng trống trong không gian mẫu một cách tình cờ. Các sampling phân tầng buộc bao phủ đồng đều bằng cách chia không gian thành các tầng và sampling từ mỗi tầng.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

sampling phân tầng luôn có variance thấp hơn hoặc bằng so với Monte Carlo tiêu chuẩn:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Ứng dụng:**
- Tích hợp số (gần như Monte Carlo)
- Training phân tách dữ liệu (đảm bảo cân bằng class trong mỗi lần gấp)
- Tầm quan trọng sampling với phân tầng (kết hợp cả hai kỹ thuật)
- NeRF (Trường bức xạ thần kinh) sử dụng sampling phân tầng dọc theo các tia máy ảnh

### Kết nối với Models khuếch tán

Khuếch tán models tạo ra hình ảnh thông qua sampling process. process chuyển tiếp thêm nhiễu Gaussian vào hình ảnh qua các bước T cho đến khi nó trở thành nhiễu thuần túy. Ngược lại, process học cách khử nhiễu, từng bước khôi phục hình ảnh gốc.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Mối liên hệ với các phương pháp trong bài học này:
- Mỗi bước khử nhiễu sử dụng thủ thuật tái tham số hóa (nhiễu mẫu, áp dụng biến đổi xác định)
- Lịch trình nhiễu {alpha_t} kiểm soát một hình thức ủ temperature
- Training sử dụng ước tính Monte Carlo để xấp xỉ ELBO (bằng chứng giới hạn dưới)
- sampling tổ tiên trong khuếch tán models là một chuỗi Markov (mỗi bước chỉ phụ thuộc vào trạng thái hiện tại)

Toàn bộ process tạo hình ảnh là sampling lặp đi lặp lại: bắt đầu từ nhiễu và ở mỗi bước, lấy mẫu một phiên bản ít nhiễu hơn một chút dựa trên model khử nhiễu đã học.

```figure
monte-carlo-pi
```

## Tự xây dựng

### Bước 1: sampling CDF đồng đều và nghịch đảo

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Tạo 10.000 mẫu theo cấp số nhân và xác minh giá trị trung bình là 1/lambda.

### Bước 2: Từ chối sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Sử dụng sampling loại bỏ để rút ra từ phân phối chuẩn bị cắt bớt. Xác minh hình dạng bằng cách biểu đồ mẫu.

### Bước 3: Tầm quan trọng sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Ước tính E[X^2] theo phân phối chuẩn bằng cách sử dụng một đề xuất thống nhất. So sánh với câu trả lời đã biết (mu^2 + sigma^2).

### Bước 4: Ước tính Monte Carlo của pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Bước 5: MCMC Metropolis-Hastings

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Mẫu từ phân phối hai phương thức (hỗn hợp của hai Gaussian). Hình dung quỹ đạo của chuỗi.

### Bước 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Bước 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Hiển thị cách temperature thay đổi phân phối đầu ra cho một tập hợp token logits.

### Bước 8: Top-k và top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Bước 9: Thủ thuật tái tham số hóa

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Chứng minh rằng gradients chảy qua mẫu được tham số lại nhưng không qua sampling trực tiếp.

### Bước 10: Kẹo cao su-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Cho thấy việc giảm temperature làm cho đầu ra tiếp cận một vector nóng như thế nào.

Triển khai đầy đủ với tất cả các hình ảnh trực quan đều `code/sampling.py`.

## Ứng dụng

Với NumPy và SciPy, các phiên bản production:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Đối với MCMC trên quy mô lớn, hãy sử dụng các thư viện chuyên dụng:
- PyMC: mô hình Bayes đầy đủ với NUTS (HMC thích ứng)
- MC: bộ lấy mẫu MCMC
- NumPyro/JAX: MCMC tăng tốc GPU

Bạn đã xây dựng những thứ này từ đầu. Bây giờ bạn biết các cuộc gọi thư viện đang làm gì.

## Bài tập

1. Triển khai sampling CDF nghịch đảo cho phân phối Cauchy. CDF là F(x) = 0,5 + arctan(x)/pi. Tạo 10.000 mẫu và vẽ biểu đồ so với PDF thực. Chú ý các đuôi nặng (giá trị cực xa tâm).

2. Sử dụng sampling từ chối để tạo mẫu từ phân phối Beta (2, 5) bằng cách sử dụng đề xuất Thống nhất (0, 1). Vẽ các mẫu được chấp nhận so với PDF Beta thực sự. Tỷ lệ chấp nhận lý thuyết là gì?

3. Ước tính tích phân của sin(x) từ 0 đến pi bằng cách sử dụng Monte Carlo với 1.000, 10.000 và 100.000 mẫu. So sánh lỗi ở từng cấp độ. Xác minh rằng lỗi có tỷ lệ là O (1/sqrt (N)).

4. Triển khai Metropolis-Hastings để lấy mẫu từ phân phối 2D p(x, y) tỷ lệ với exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Vẽ các mẫu và quỹ đạo chuỗi. Thử nghiệm với các độ lệch chuẩn đề xuất khác nhau.

5. Xây dựng một bản demo tạo văn bản hoàn chỉnh: cho một từ vựng gồm 10 từ với logits, tạo chuỗi 20 tokens bằng cách sử dụng (a) tham lam, (b) temperature = 0,7, (c) top-k = 3, (d) top-p = 0,9. So sánh sự đa dạng của đầu ra trong 5 lần chạy.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Sampling | "Rút các giá trị ngẫu nhiên" | Tạo ra các giá trị theo phân phối xác suất. Cơ chế đằng sau tất cả các AI tạo ra |
| Phân phối đồng đều | "Tất cả đều có khả năng như nhau" | Mọi giá trị trong [a, b] có mật độ xác suất bằng nhau 1/(b-a). Điểm khởi đầu cho tất cả các phương thức sampling |
| CDF nghịch đảo | "Biến đổi xác suất" | F_inverse(U) chuyển đổi một mẫu đồng nhất thành một mẫu từ bất kỳ phân phối nào có CDF đã biết. Chính xác và hiệu quả |
| Từ chối sampling | "Đề xuất và accept/reject" | Tạo từ một đề xuất đơn giản, chấp nhận với tỷ lệ xác suất tỷ lệ target/proposal. Chính xác nhưng lãng phí mẫu |
| Tầm quan trọng sampling | "Cân lại mẫu" | Ước tính kỳ vọng theo p(x) bằng cách sử dụng các mẫu từ q(x) bằng cách tính trọng số mỗi mẫu bằng p(x)/q(x). Core to PPO trong RL |
| Monte Carlo | "Mẫu ngẫu nhiên trung bình" | Tích phân gần đúng dưới dạng trung bình mẫu. Lỗi O (1/sqrt (N)) bất kể kích thước |
| MCMC | "Đi bộ ngẫu nhiên hội tụ" | Xây dựng một chuỗi Markov có phân bố cố định là mục tiêu. Metropolis-Hastings là thuật toán nền tảng |
| Đô thị-Hastings | "Chấp nhận lên dốc, đôi khi xuống dốc" | Đề xuất di chuyển, chấp nhận dựa trên tỷ lệ mật độ. Cân bằng chi tiết đảm bảo hội tụ phân phối mục tiêu |
| Gibbs sampling | "Mỗi lần một biến" | Cập nhật từng biến từ phân phối có điều kiện của nó giữ các biến khác cố định. Tỷ lệ chấp nhận 100% |
| Temperature | "Núm tự tin" | Chia logits cho T trước softmax. T<1 sắc nét (tự tin hơn), T>1 làm phẳng (đa dạng hơn) |
| Top-k sampling | "Giữ k tốt nhất" | Loại bỏ tất cả trừ k xác suất cao nhất tokens, chuẩn hóa, mẫu. Kích thước bộ ứng cử viên cố định |
| Hạt nhân sampling (top-p) | "Giữ lại những điều có thể xảy ra" | Giữ tập tokens nhỏ nhất có xác suất tích lũy vượt quá p. Kích thước bộ ứng viên thích ứng |
| Thủ thuật tái tham số hóa | "Di chuyển ngẫu nhiên ra bên ngoài" | Viết z = mu + sigma * epsilon trong đó epsilon ~ N(0,1). Làm cho sampling có thể phân biệt. Cần thiết cho VAE training |
| Gumbel-Softmax | "sampling phân loại mềm" | Xấp xỉ có thể phân biệt với sampling phân loại bằng cách sử dụng nhiễu Gumbel + softmax với temperature |
| Phân tầng sampling | "Bảo hiểm bắt buộc" | Chia không gian mẫu thành các tầng, mẫu từ mỗi tầng. Luôn variance thấp hơn Monte Carlo ngây thơ |
| Burn-in | "Thời gian khởi động" | Các mẫu MCMC ban đầu bị loại bỏ trước khi chuỗi đạt đến sự phân bố cố định của nó |
| Cân chi tiết | "Điều kiện khả năng đảo ngược" | p(x) * T(x->y) = p(y) * T(y->x). Điều kiện đủ để p trở thành phân phối tĩnh của chuỗi Markov |
| Khuếch tán sampling | "Khử nhiễu lặp đi lặp lại" | Tạo dữ liệu bằng cách bắt đầu từ nhiễu và áp dụng các bước khử nhiễu đã học. Mỗi bước là một hoạt động sampling có điều kiện |

## Đọc thêm

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - hướng dẫn chi tiết về nền tảng MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - giấy Gumbel-Softmax chính hãng
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - hạt nhân (top-p) sampling giấy
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Bài báo VAE giới thiệu thủ thuật tái tham số hóa
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM kết nối sampling với tạo hình ảnh
