# Thống kê cho Machine Learning

> Thống kê là cách bạn biết liệu model của bạn có thực sự hoạt động hay chỉ may mắn.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 06 (Xác suất và Phân phối), 07 (Định lý Bayes)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Tính toán số liệu thống kê mô tả, tương quan Pearson/Spearman và ma trận hiệp phương sai từ đầu
- Thực hiện các bài kiểm tra giả thuyết (t-test, chi-squared) và giải thích các giá trị p và khoảng tin cậy một cách chính xác
- Sử dụng lấy mẫu lại bootstrap để xây dựng khoảng tin cậy cho bất kỳ chỉ số nào mà không cần giả định phân phối
- Phân biệt ý nghĩa thống kê với ý nghĩa thực tế bằng cách sử dụng các thước đo kích thước hiệu ứng

## Vấn đề

Bạn đã huấn luyện hai models. Model A đạt 0,87 điểm trong bộ bài kiểm tra của bạn. Model điểm B 0,89. Bạn triển khai Model B. Ba tuần sau, các chỉ số production tệ hơn trước. Chuyện gì đã xảy ra?

Model B không thực sự vượt trội hơn Model A. Sự khác biệt 0,02 là nhiễu. Bộ kiểm tra của bạn quá nhỏ hoặc variance quá cao hoặc cả hai. Bạn shipped sự ngẫu nhiên được ăn mặc như một sự cải thiện.

Điều này xảy ra liên tục. Thay đổi bảng xếp hạng Kaggle. Giấy tờ không sao chép. A/B thử nghiệm tuyên bố người chiến thắng dựa trên vài trăm mẫu. Nguyên nhân gốc rễ luôn giống nhau: ai đó đã bỏ qua số liệu thống kê.

Thống kê cung cấp cho bạn các công cụ để phân biệt tín hiệu với nhiễu. Nó cho bạn biết khi nào sự khác biệt là có thật, bạn nên tự tin như thế nào và bạn cần bao nhiêu dữ liệu trước khi bạn có thể tin tưởng vào kết quả. Mỗi ML pipeline, mỗi model so sánh, mọi thử nghiệm đều cần thống kê. Nếu không có nó, bạn đang đoán.

## Khái niệm

### Thống kê mô tả: Tóm tắt dữ liệu của bạn

Trước khi model bất cứ điều gì, bạn cần biết dữ liệu của mình trông như thế nào. Thống kê mô tả nén một dataset thành một vài số nắm bắt được hình dạng của nó.

**Các biện pháp về xu hướng trung tâm** trả lời "ở đâu?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Giá trị trung bình là điểm cân bằng. Trung vị là mốc nửa chừng. Khi chúng phân kỳ, phân phối của bạn bị lệch. Phân phối thu nhập có nghĩa là >> trung bình (nghiêng phải từ các tỷ phú). Phân phối Loss trong training thường có trung bình << trung bình (lệch trái so với các mẫu dễ).

**Các biện pháp lây lan** trả lời "dữ liệu phân tán như thế nào?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Phần trăm** chia dữ liệu được sắp xếp thành 100 phần bằng nhau. Phân vị thứ 25 (Q1) có nghĩa là 25% giá trị giảm xuống dưới điểm này. Phân vị thứ 50 là giá trị trung bình. Phân vị thứ 75 là Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

Trong ML, bạn quan tâm đến phân vị cho độ trễ inference, phân phối độ tin cậy dự đoán và hiểu phân phối lỗi. Một model có lỗi trung bình thấp nhưng lỗi P99 khủng khiếp có thể vô dụng đối với các ứng dụng quan trọng về an toàn.

**Thống kê mẫu so với dân số.** Khi tính variance từ mẫu, chia cho (n-1) thay vì n. Đây là sự sửa chữa của Bessel. Nó bù đắp cho thực tế là giá trị trung bình mẫu của bạn không phải là giá trị trung bình dân số thực. Với n trong mẫu số, bạn đánh giá thấp variance thực sự một cách có hệ thống. Với (n-1), ước tính là không thiên vị.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Trong thực tế: nếu n lớn (hàng nghìn mẫu), sự khác biệt là không đáng kể. Nếu n nhỏ (hàng chục mẫu), điều đó rất quan trọng.

### Tương quan: Cách các biến di chuyển cùng nhau

Tương quan đo lường sức mạnh và hướng của mối quan hệ tuyến tính giữa hai biến.

**Hệ số tương quan Pearson** đo lường liên kết tuyến tính:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson giả định mối quan hệ là tuyến tính và cả hai biến đều được phân phối gần như bình thường. Nó nhạy cảm với các ngoại lệ. Một điểm cực trị duy nhất có thể kéo r từ 0,1 đến 0,9.

**Tương quan cấp bậc Spearman** đo lường mối liên hệ đơn điệu:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Khi nào sử dụng mỗi:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Quy tắc vàng: **tương quan không ngụ ý nhân quả. Doanh số bán kem và tử vong do đuối nước có tương quan với nhau vì cả hai đều tăng vào mùa hè. accuracy model của bạn và số lượng parameters có tương quan với nhau, nhưng việc thêm parameters không tự động cải thiện accuracy (xem: overfitting).

### Ma trận hiệp phương sai

Hiệp phương sai giữa hai biến đo lường cách chúng thay đổi cùng nhau:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Đối với d features, ma trận hiệp phương sai C là ma trận d x d trong đó C [i] [j] = Cov (feature_i, feature_j). Các mục chéo C[i][i] là phương sai của mỗi feature.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Kết nối với PCA.** PCA riêng phân hủy ma trận hiệp phương sai. Các vectơ riêng là các thành phần chính (hướng của variance tối đa). Các giá trị riêng cho bạn biết mỗi thành phần nắm bắt được bao nhiêu variance. Đây chính xác là những gì Bài 10 đã đề cập, nhưng bây giờ bạn thấy tại sao ma trận hiệp phương sai là thứ đúng đắn để phân hủy: nó mã hóa tất cả các mối quan hệ tuyến tính theo cặp trong dữ liệu của bạn.

**Kết nối với tương quan.** Ma trận tương quan là ma trận hiệp phương sai của các biến chuẩn hóa (mỗi biến chia cho độ lệch chuẩn của nó). Tương quan chuẩn hóa hiệp phương sai để tất cả các giá trị rơi vào [-1, 1].

### Kiểm tra giả thuyết

Kiểm tra giả thuyết là một framework để đưa ra quyết định trong điều kiện không chắc chắn. Bạn bắt đầu với một yêu cầu, thu thập dữ liệu và xác định xem dữ liệu có phù hợp với xác nhận quyền sở hữu hay không.

**Thiết lập:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**Giá trị p** là xác suất nhìn thấy dữ liệu cực đoan như những gì bạn quan sát được, giả sử H0 là đúng. KHÔNG phải xác suất H0 là đúng. Đây là sự hiểu lầm phổ biến nhất trong thống kê.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Khoảng tin cậy** đưa ra một loạt các giá trị hợp lý cho một parameter:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Chiều rộng của khoảng tin cậy cho bạn biết về precision. Khoảng thời gian rộng có nghĩa là độ không chắc chắn cao. Khoảng thời gian hẹp có nghĩa là ước tính của bạn chính xác (nhưng không nhất thiết phải chính xác, nếu dữ liệu của bạn bị sai lệch).

### Bài kiểm tra t

T-test so sánh các phương tiện. Có một số hương vị.

**Kiểm tra t một mẫu: **trung bình tổng thể có khác với giá trị giả thuyết không?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Thử nghiệm t hai mẫu (độc lập):** Hai nhóm có nghĩa là khác nhau không?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Kiểm tra t được ghép nối: **khi các phép đo đến theo cặp (cùng một model được đánh giá trên cùng một phân tách dữ liệu):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Trong ML, kiểm tra t được ghép đôi là phổ biến: bạn chạy cả hai models trên cùng 10 lần xác thực chéo và so sánh điểm số của chúng theo cặp.

### Kiểm tra chi bình phương

Thử nghiệm chi bình phương kiểm tra xem các tần số quan sát được có khớp với tần số dự kiến hay không. Hữu ích cho dữ liệu phân loại.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Kiểm tra ML Models

A/B kiểm thử trong ML không giống như kiểm thử web A/B. So sánh Model có những thách thức cụ thể:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Thủ tục:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Ý nghĩa thống kê vs Ý nghĩa thực tế

Một kết quả có thể có ý nghĩa thống kê nhưng thực tế là vô nghĩa. Với đủ dữ liệu, ngay cả một sự khác biệt nhỏ cũng trở nên có ý nghĩa thống kê.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Kích thước hiệu ứng** định lượng mức độ chênh lệch lớn, không phụ thuộc vào kích thước mẫu:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Luôn báo cáo cả giá trị p và kích thước hiệu ứng. Giá trị p cho bạn biết liệu sự khác biệt có thật hay không. Kích thước hiệu ứng cho bạn biết liệu nó có quan trọng hay không.

### Nhiều vấn đề so sánh

Khi bạn kiểm tra nhiều giả thuyết, một số giả thuyết sẽ "đáng kể" một cách tình cờ. Nếu bạn kiểm tra 20 thứ ở alpha = 0,05, bạn mong đợi 1 dương tính giả ngay cả khi không có gì là thật.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Hiệu chỉnh Bonferroni: **chia alpha cho số lần kiểm tra.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML, điều này quan trọng khi bạn so sánh một model trên nhiều chỉ số, thử nghiệm nhiều cấu hình hyperparameter hoặc đánh giá trên nhiều datasets.

### Phương pháp Bootstrap

Bootstrapping ước tính sự phân phối sampling của một thống kê bằng cách lấy mẫu lại dữ liệu của bạn bằng cách thay thế. Không cần giả định về phân phối cơ bản.

**Thuật toán:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Khoảng tin cậy Bootstrap (phương pháp phần trăm):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Tại sao bootstrap lại quan trọng đối với ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap để so sánh model:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Điều này mạnh mẽ hơn so với kiểm tra t cặp vì nó không đưa ra giả định phân phối.

### Kiểm tra tham số so với không tham số

**Kiểm tra tham số** giả định một phân phối cụ thể (thường là bình thường):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Thử nghiệm phi tham số** không đưa ra giả định phân phối:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Khi nào nên sử dụng phi tham số:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Khi nào sử dụng tham số:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

Trong các thử nghiệm ML, bạn thường có n nhỏ (5 hoặc 10 nếp gấp xác thực chéo), vì vậy các bài kiểm tra phi tham số như Wilcoxon signed-rank thường thích hợp hơn t-test.

### Định lý giới hạn trung tâm: Ý nghĩa thực tế

CLT cho biết sự phân bố của các phương tiện mẫu tiếp cận phân phối chuẩn khi n tăng lên, bất kể phân bố dân số cơ bản.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Tại sao điều này lại quan trọng đối với ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Những gì CLT KHÔNG làm:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Những sai lầm thống kê phổ biến trong các bài báo ML

1. **Thử nghiệm trên bộ training.** Đảm bảo overfitting. Luôn giữ dữ liệu mà model không bao giờ nhìn thấy trong quá trình training.

2. **Không có khoảng tin cậy.** Báo cáo một số accuracy duy nhất mà không có độ không chắc chắn làm cho kết quả không thể tái tạo và không thể xác minh được.

3. **Bỏ qua nhiều so sánh.** Thử nghiệm 50 cấu hình và báo cáo cấu hình tốt nhất mà không cần chỉnh sửa sẽ làm tăng tỷ lệ dương tính giả.

4. **Ý nghĩa thống kê và thực tiễn gây nhầm lẫn.** Giá trị p 0,001 trên sự cải thiện 0,01% accuracy không có ý nghĩa.

5. **Sử dụng accuracy trên dữ liệu mất cân bằng.** 99% accuracy trên dataset có 99% class âm có nghĩa là model không học được gì. Sử dụng precision, recall, F1 hoặc AUC.

6. **Chỉ báo cáo chỉ số mà model của bạn chiến thắng. Đánh giá trung thực báo cáo tất cả các chỉ số liên quan.

7. **Rò rỉ thông tin qua train/test phân tách.** Chuẩn hóa trước khi phân tách hoặc sử dụng dữ liệu trong tương lai để dự đoán quá khứ.

8. **Các bộ thử nghiệm nhỏ không có ước tính variance.** Đánh giá trên 100 mẫu và tuyên bố cải thiện 2% là nhiễu chứ không phải tín hiệu.

9. **Giả định độc lập khi dữ liệu không độc lập.** Hình ảnh y tế từ cùng một bệnh nhân, nhiều câu từ cùng một tài liệu. Các quan sát trong một nhóm có mối tương quan với nhau.

10. **P-hacking.** Thử các bài kiểm tra, tập hợp con hoặc tiêu chí loại trừ khác nhau cho đến khi bạn nhận được p < 0,05. Kết quả là một artifact của tìm kiếm.

## Xây dựng nó

Bạn sẽ triển khai:

1. **Thống kê mô tả từ đầu** (trung bình, trung bình, chế độ, độ lệch chuẩn, phần trăm, IQR)
2. **Hàm tương quan** (Pearson và Spearman, với ma trận hiệp phương sai)
3. **Kiểm tra giả thuyết** (kiểm tra t một mẫu, kiểm tra t hai mẫu, kiểm tra chi bình phương)
4. **Khoảng tin cậy Bootstrap** (đối với bất kỳ thống kê nào, không cần giả định)
5. **A/B trình mô phỏng kiểm tra **(tạo dữ liệu, kiểm tra, kiểm tra lỗi Loại I và Loại II)
6. **Bản demo ý nghĩa thống kê và thực tế** (cho thấy n lớn làm cho mọi thứ trở nên "có ý nghĩa")

Tất cả từ đầu, chỉ sử dụng `math` và `random`. Không numpy, không có scipy.

## Thuật ngữ chính

| Thuật ngữ | Định nghĩa |
|---|---|
| Trung bình | Tổng các giá trị chia cho số lượng. Nhạy cảm với các ngoại lệ. |
| Trung bình | Giá trị giữa của dữ liệu được sắp xếp. Mạnh mẽ đến các ngoại lệ. |
| Độ lệch chuẩn | Căn bậc hai của variance. Các biện pháp trải rộng theo đơn vị ban đầu. |
| Phân vị | Giá trị dưới đó một tỷ lệ phần trăm dữ liệu nhất định giảm. |
| Chỉ số IQR | Phạm vi liên tứ vị. Q3 trừ Q1. Mức chênh lệch của mức trung bình là 50%. |
| Tương quan Pearson | Đo lường liên kết tuyến tính giữa hai biến. Phạm vi [-1, 1]. |
| Tương quan Spearman | Đo lường sự liên kết đơn điệu bằng cách sử dụng cấp bậc. |
| Ma trận hiệp phương sai | Ma trận hiệp phương sai theo cặp giữa tất cả features. |
| Giả thuyết không | Giả định mặc định không có tác dụng hoặc không có sự khác biệt. |
| giá trị p | Xác suất dữ liệu cực đoan này với giả thuyết không là đúng. |
| Khoảng tin cậy | Phạm vi giá trị hợp lý cho một parameter ở một mức độ tin cậy nhất định. |
| Kiểm tra chữ T | Kiểm tra xem các phương tiện có khác nhau đáng kể hay không. Sử dụng phân phối t. |
| Kiểm tra chi bình phương | Kiểm tra xem tần số quan sát có khác với tần số dự kiến hay không. |
| Kích thước hiệu ứng | Độ lớn của sự khác biệt, không phụ thuộc vào kích thước mẫu. D của Cohen là phổ biến. |
| Hiệu chỉnh Bonferroni | Chia ngưỡng ý nghĩa cho số lần xét nghiệm để kiểm soát dương tính giả. |
| Dây khởi động | Lấy mẫu lại với thay thế để ước tính phân phối sampling. |
| Lỗi loại I | Dương tính giả. Từ chối H0 khi nó là đúng. |
| Lỗi loại II | Âm tính giả. Không từ chối H0 khi nó sai. |
| Sức mạnh thống kê | Xác suất từ chối chính xác H0 sai. Công suất = 1 trừ tỷ lệ lỗi Loại II. |
| Định lý giới hạn trung tâm | Mẫu có nghĩa là hội tụ về phân phối chuẩn khi kích thước mẫu tăng lên. |
| Kiểm tra tham số | Giả định một phân phối cụ thể cho dữ liệu (thường là bình thường). |
| Kiểm tra phi tham số | Không đưa ra giả định phân phối. Hoạt động trên cấp bậc hoặc dấu hiệu. |
