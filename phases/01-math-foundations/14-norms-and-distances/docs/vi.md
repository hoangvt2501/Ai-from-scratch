# Định mức và khoảng cách

> Hàm khoảng cách của bạn xác định ý nghĩa của "tương tự". Chọn sai và mọi thứ xuôi dòng bị hỏng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1, Bài 01 (Trực giác đại số tuyến tính), 02 (Vectors, Ma trận & Phép toán)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Triển khai các chức năng L1, L2, cosine, Mahalanobis, Jaccard và chỉnh sửa khoảng cách từ đầu
- Chọn chỉ số khoảng cách thích hợp cho một nhiệm vụ ML nhất định và giải thích lý do tại sao các lựa chọn thay thế không thành công
- Kết nối các chuẩn L1 và L2 với chính quy hóa LASSO và Ridge và các vùng ràng buộc hình học của chúng
- Chứng minh cách cùng một dataset tạo ra các hàng xóm gần nhất khác nhau theo các chỉ số khác nhau

## Vấn đề

Bạn có hai vectors. Có lẽ chúng là từ embeddings. Có thể chúng là hồ sơ người dùng. Có thể chúng là mảng pixel. Bạn cần biết: họ gần gũi như thế nào?

Câu trả lời hoàn toàn phụ thuộc vào hàm khoảng cách bạn chọn. Hai điểm dữ liệu có thể là hàng xóm gần nhất trong một số liệu và cách xa nhau dưới một số liệu khác. Bộ phân loại KNN, công cụ đề xuất, cơ sở dữ liệu vector, thuật toán phân cụm, hàm loss của bạn - tất cả đều phụ thuộc vào lựa chọn này. Làm sai và model của bạn sẽ tối ưu hóa cho điều sai.

Không có khoảng cách tốt nhất phổ quát. L2 hoạt động cho dữ liệu không gian. Sự tương đồng cosin chiếm ưu thế NLP. Jaccard xử lý các bộ. Chỉnh sửa khoảng cách xử lý chuỗi. Mahalanobis giải thích cho các mối tương quan. Wasserstein di chuyển khối lượng xác suất. Mỗi người mã hóa một giả định khác nhau về ý nghĩa của "tương tự".

Bài học này xây dựng mọi hàm khoảng cách chính từ đầu, cho bạn biết khi nào mỗi hàm là công cụ phù hợp và minh họa cách cùng một dữ liệu tạo ra các hàm lân cận gần nhất hoàn toàn khác nhau tùy thuộc vào chỉ số bạn sử dụng.

## Khái niệm

### Định mức: đo vector độ lớn

Một định mức đo "kích thước" của một vector. Mọi hàm khoảng cách giữa hai vectors có thể được viết dưới dạng chuẩn của sự khác biệt của chúng: d(a, b) = ||a - b||. Vì vậy, hiểu các chuẩn mực là hiểu khoảng cách.

### L1 Norm (khoảng cách Manhattan)

Định mức L1 tổng các giá trị tuyệt đối của tất cả các thành phần.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nó được gọi là khoảng cách Manhattan vì nó đo quãng đường bạn đi bộ trên lưới thành phố, nơi bạn chỉ có thể di chuyển dọc theo các trục. Không có đường chéo.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Khi nào nên sử dụng L1:
- High-dimensional dữ liệu thưa thớt (features văn bản, mã hóa một nóng)
- Khi bạn muốn sự mạnh mẽ đến các ngoại lệ (một sự khác biệt lớn duy nhất không chiếm ưu thế)
- Feature vấn đề lựa chọn (chính quy hóa L1 thúc đẩy sự thưa thớt)

Kết nối với chính quy hóa L1 (Lasso): thêm ||w||_1 vào hàm loss của bạn sẽ phạt tổng các giá trị trọng số tuyệt đối. Điều này đẩy trọng lượng nhỏ về chính xác bằng không, thực hiện lựa chọn feature tự động. Hình phạt L1 tạo ra các vùng ràng buộc hình kim cương trong không gian trọng lượng và các góc của kim cương nằm trên các trục nơi một số trọng lượng bằng không.

Kết nối với các hàm loss: Sai số tuyệt đối trung bình (MAE) là khoảng cách L1 trung bình giữa các dự đoán và mục tiêu. Nó phạt tất cả các lỗi một cách tuyến tính, làm cho nó mạnh mẽ đối với các ngoại lệ so với MSE.

### L2 Norm (khoảng cách Euclid)

Định mức L2 là khoảng cách đường thẳng. Căn bậc hai của tổng các thành phần bình phương.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Đây là khoảng cách bạn học được trong class hình học. Pythagoras trong n chiều.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Khi nào nên sử dụng L2:
- Dữ liệu liên tục chiều từ thấp đến trung bình
- Khi các thang đo feature có thể so sánh được
- Khoảng cách vật lý (dữ liệu không gian, số đọc cảm biến)
- Sự tương đồng của hình ảnh ở cấp độ pixel

Kết nối với chính quy hóa L2 (Ridge): thêm ||w||_2^2 đối với hàm loss của bạn sẽ phạt trọng lượng lớn. Không giống như L1, nó không đẩy trọng lượng về không. Nó thu nhỏ tất cả các trọng lượng về không theo tỷ lệ. Hình phạt L2 tạo ra các vùng ràng buộc hình tròn, vì vậy không có góc trên trục. Trọng lượng nhỏ nhưng hiếm khi chính xác bằng không.

Kết nối với các hàm loss: Sai số bình phương trung bình (MSE) là giá trị trung bình của khoảng cách L2 bình phương. Bình phương phạt các lỗi lớn nặng hơn những lỗi nhỏ.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Định mức Lp: gia đình chung

L1 và L2 là những trường hợp đặc biệt của định mức Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Các giá trị khác nhau của p tạo ra các "quả bóng đơn vị" có hình dạng khác nhau (tập hợp tất cả các điểm ở khoảng cách 1 từ điểm gốc):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm (khoảng cách Chebyshev)

Khi p tiếp cận vô cực, định mức Lp hội tụ đến thành phần tuyệt đối tối đa.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Khoảng cách giữa hai điểm được xác định bởi chiều duy nhất nơi chúng khác nhau nhiều nhất. Tất cả các kích thước khác đều bị bỏ qua.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Khi nào nên sử dụng L-infinity:
- Khi độ lệch trong trường hợp xấu nhất trong bất kỳ chiều đơn lẻ nào quan trọng
- Bảng trò chơi (một vị vua trong cờ vua di chuyển trong L-vô cực: một bước theo bất kỳ hướng nào có giá 1)
- Dung sai sản xuất (mọi kích thước phải nằm trong thông số kỹ thuật)

### Độ tương đồng cosin và khoảng cách cosin

Sự tương đồng cosin đo góc giữa hai vectors, bỏ qua độ lớn của chúng.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Nó nằm trong khoảng từ -1 (hướng ngược lại) đến +1 (cùng hướng). vectors vuông góc có độ tương tự cosin 0.

Khoảng cách cosin chuyển đổi nó thành khoảng cách: cosine_distance = 1 - cosine_similarity. Điều này nằm trong khoảng từ 0 (cùng hướng) đến 2 (hướng ngược lại).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Tại sao cosin chiếm ưu thế NLP và embeddings: trong văn bản, độ dài tài liệu không nên ảnh hưởng đến sự tương đồng. Một tài liệu về mèo dài gấp đôi so với một tài liệu khác về mèo vẫn nên "tương tự". Sự tương đồng cosin bỏ qua độ lớn (chiều dài) và chỉ quan tâm đến hướng. Hai tài liệu có cùng phân phối từ nhưng độ dài khác nhau chỉ theo cùng một hướng và có được độ tương đồng cosin 1.0.

Khi nào nên sử dụng sự tương tự cosin:
- Sự giống nhau của văn bản (TF-IDF vectors, embeddings từ, câu embeddings)
- Bất kỳ miền nào có độ lớn là nhiễu và hướng là tín hiệu
- Hệ thống đề xuất (tùy chọn người dùng vectors)
- Tìm kiếm Embedding (vector cơ sở dữ liệu hầu như luôn sử dụng cosin hoặc tích chấm)

### Điểm giống sản phẩm so với Cosine

Tích chấm của hai vectors là:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Độ tương tự cosin là tích chấm được chuẩn hóa theo cả hai cấp độ. Khi cả hai vectors đã được chuẩn hóa đơn vị (độ lớn = 1), tích chấm và sự tương đồng cosin giống hệt nhau.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Khi chúng khác nhau: sản phẩm chấm bao gồm thông tin độ lớn. Một vector có độ lớn lớn hơn sẽ có điểm sản phẩm chấm cao hơn. Điều này quan trọng trong một số hệ thống truy xuất mà bạn muốn các mặt hàng "phổ biến" xếp hạng cao hơn. Độ lớn hoạt động như một tín hiệu chất lượng hoặc tầm quan trọng ngầm.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

Trong thực tế:
- Sử dụng độ tương tự cosin khi bạn muốn có sự tương đồng về hướng thuần túy
- Sử dụng sản phẩm chấm khi độ lớn mang thông tin có ý nghĩa
- Nhiều cơ sở dữ liệu vector (Pinecone, Weaviate, Qdrant) cho phép bạn lựa chọn giữa chúng
- Nếu embeddings của bạn được chuẩn hóa L2, sự lựa chọn không quan trọng

### Khỏang cách Mahalanobis

Khoảng cách Euclid đối xử với tất cả các chiều như nhau. Nhưng nếu features của bạn có tương quan hoặc có thang đo khác nhau, L2 sẽ cho kết quả sai lệch.

Khoảng cách Mahalanobis giải thích cho cấu trúc hiệp phương sai của dữ liệu.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

trong đó S là ma trận hiệp phương sai của dữ liệu.

Theo trực giác: Khoảng cách Mahalanobis trước tiên khử tương quan và chuẩn hóa dữ liệu (làm trắng), sau đó tính toán khoảng cách L2 trong không gian biến đổi đó. Nếu S là ma trận nhận dạng (không tương quan, đơn vị variance features), khoảng cách Mahalanobis giảm xuống khoảng cách Euclid.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Khi nào nên sử dụng khoảng cách Mahalanobis:
- Phát hiện ngoại lệ (các điểm có khoảng cách Mahalanobis lớn so với giá trị trung bình là giá trị ngoại lệ)
- Phân loại khi features có thang đo và tương quan khác nhau
- Khi bạn có đủ dữ liệu để ước tính ma trận hiệp phương sai đáng tin cậy
- Kiểm soát chất lượng trong sản xuất (giám sát process đa biến)

### Jaccard Similarity (dành cho bộ)

Các thước đo tương tự của Jaccard chồng chéo giữa hai bộ.

```
J(A, B) = |A intersect B| / |A union B|
```

Nó nằm trong khoảng từ 0 (không chồng chéo) đến 1 (các tập giống hệt nhau). Khoảng cách Jaccard = 1 - Sự tương đồng của Jaccard.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Khi nào sử dụng Jaccard:
- So sánh các bộ thẻ, danh mục hoặc features
- Tài liệu tương tự dựa trên sự hiện diện của từ (không phải tần suất)
- Phát hiện gần trùng lặp (Xấp xỉ MinHash của Jaccard)
- So sánh feature vectors nhị phân (dữ liệu presence/absence)
- Đánh giá phân đoạn models (Giao lộ qua Liên minh = Jaccard)

### Chỉnh sửa khoảng cách (Khoảng cách Levenshtein)

Khoảng cách chỉnh sửa đếm số lượng thao tác đơn ký tự tối thiểu cần thiết để chuyển đổi chuỗi này sang chuỗi khác. Các thao tác là: chèn, xóa hoặc thay thế.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Được tính toán bằng lập trình động. Điền vào ma trận trong đó mục nhập (i, j) là khoảng cách chỉnh sửa giữa các ký tự i đầu tiên của chuỗi A và các ký tự j đầu tiên của chuỗi B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Khi nào nên sử dụng khoảng cách chỉnh sửa:
- Kiểm tra chính tả và sửa lỗi
- Trình tự DNA alignment (với các phép toán có trọng số)
- Khớp chuỗi mờ
- Loại bỏ trùng lặp dữ liệu văn bản lộn xộn

### Phân kỳ KL (không phải khoảng cách, nhưng được sử dụng như một)

Phân kỳ KL đo lường sự khác biệt của một phân phối xác suất với phân phối khác. Được đề cập trong Bài 09, nhưng nó thuộc về cuộc thảo luận này vì mọi người sử dụng nó như một "khoảng cách" mặc dù nó không phải là một khoảng cách.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Thuộc tính quan trọng: Phân kỳ KL KHÔNG đối xứng.

```
D_KL(P || Q) != D_KL(Q || P)
```

Điều này có nghĩa là nó không đáp ứng yêu cầu cơ bản của số liệu khoảng cách. Nó cũng không thỏa mãn bất đẳng thức tam giác. Đó là một sự phân kỳ, không phải là một khoảng cách.

Chuyển tiếp KL (D_KL(P || Q)) là "mean-seeking": Q cố gắng bao quát tất cả các chế độ của P.
Đảo ngược KL (D_KL(Q || P)) là "tìm kiếm chế độ": Q tập trung vào một chế độ duy nhất của P.

Khi bạn thấy phân kỳ KL:
- VAE (thuật ngữ KL trong ELBO đẩy phân phối tiềm ẩn về phía prior)
- distillation kiến thức (học sinh cố gắng khớp với sự phân phối của giáo viên)
- RLHF (hình phạt KL giữ fine-tuned model gần model cơ sở)
- Policy gradient phương thức (hạn chế policy cập nhật)

### Khoảng cách Wasserstein (Khoảng cách của Earth Mover)

Khoảng cách Wasserstein đo lường "công việc" tối thiểu cần thiết để chuyển đổi phân phối xác suất này sang phân phối xác suất khác. Hãy nghĩ về nó như: nếu một phân phối là một đống đất và phân phối kia là một cái hố, bạn phải di chuyển bao nhiêu bụi bẩn và bao xa?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Đối với phân phối 1D, nó đơn giản hóa thành tích phân của sự khác biệt tuyệt đối của các hàm phân phối tích lũy:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Tại sao Wasserstein lại quan trọng:
- Nó là một số liệu thực sự (đối xứng, thỏa mãn bất đẳng thức tam giác)
- Nó cung cấp gradients ngay cả khi các phân phối không trùng lặp (phân kỳ KL đi đến vô cực)
- Đặc tính này khiến nó trở thành trung tâm của Wasserstein GAN (WGAN), giải quyết training sự mất ổn định của GAN ban đầu

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Khi nào nên sử dụng Wasserstein:
- GAN training (WGAN, WGAN-GP)
- So sánh các bản phân phối có thể không trùng lặp
- Vấn đề transport tối ưu
- Truy xuất hình ảnh (so sánh biểu đồ màu)

### Tại sao các nhiệm vụ khác nhau cần khoảng cách khác nhau

| Nhiệm vụ | Khoảng cách tốt nhất | Tại sao |
|------|--------------|-----|
| Sự giống nhau của văn bản | Cosin | Độ lớn là nhiễu, hướng là ý nghĩa |
| So sánh pixel hình ảnh | L2 · | Các mối quan hệ không gian quan trọng, features là quy mô có thể so sánh được |
| features độ mờ cao thưa thớt | L1 | Mạnh mẽ, không khuếch đại sự khác biệt lớn hiếm hoi |
| Đặt chồng chéo (thẻ, danh mục) | Jaccard | Dữ liệu được đặt giá trị tự nhiên, không phải vectơ |
| Khớp chuỗi | Chỉnh sửa khoảng cách | Ánh xạ hoạt động với trực giác chỉnh sửa của con người |
| Phát hiện ngoại lệ | Mahalanobis | Tính đến feature tương quan và thang đo |
| So sánh các bản phân phối | Phân kỳ KL | Đo lường thông tin bị mất bằng cách sử dụng Q thay vì P |
| GAN training | Wasserstein | Cung cấp gradients ngay cả khi các bản phân phối không trùng lặp |
| Embeddings (vector DB) | Cosin hoặc sản phẩm chấm | Embeddings được huấn luyện để mã hóa ý nghĩa theo hướng |
| Khuyến nghị | Sản phẩm chấm | Độ lớn có thể mã hóa mức độ phổ biến hoặc độ tin cậy |
| Trình tự DNA | Khoảng cách chỉnh sửa có trọng số | Chi phí thay thế khác nhau tùy theo cặp nucleotide |
| QC sản xuất | L-vô cực | Độ lệch trong trường hợp xấu nhất trong bất kỳ khía cạnh nào cũng quan trọng |

### Kết nối với các chức năng Loss

Loss hàm là hàm khoảng cách được áp dụng cho dự đoán so với mục tiêu.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Kết nối với chính quy hóa

Chính quy hóa thêm một hình phạt định mức trên trọng số vào hàm loss.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Tại sao L1 tạo ra sự thưa thớt nhưng L2 thì không: hình dung vùng ràng buộc trong không gian trọng lượng 2D. L1 là một viên kim cương, L2 là một hình tròn. Các đường viền của hàm loss (hình elip) rất có thể chạm vào viên kim cương ở một góc, trong đó một trọng lượng bằng không. Chúng chạm vào vòng tròn tại một điểm nhẵn, trong đó cả hai trọng lượng đều không bằng không.

### Tìm kiếm hàng xóm gần nhất

Mỗi hàm khoảng cách ngụ ý một vấn đề tìm kiếm lân cận gần nhất: cho một điểm truy vấn, hãy tìm các điểm gần nhất trong một dataset.

Tìm kiếm lân cận gần nhất chính xác là O (n * d) cho mỗi truy vấn trong một dataset n điểm có chiều d. Đối với datasets lớn, điều này là quá chậm.

Thuật toán Hàng xóm gần nhất (ANN) giao dịch một lượng nhỏ accuracy để tăng tốc độ lớn:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) là thuật toán thống trị trong cơ sở dữ liệu vector hiện đại. Nó xây dựng một biểu đồ nhiều lớp trong đó mỗi nút kết nối với các hàng xóm gần nhất của nó. Tìm kiếm bắt đầu ở lớp trên cùng (nhảy xa, thưa thớt) và đi xuống lớp dưới cùng (nhảy ngắn, dày đặc).

```figure
norm-unit-balls
```

## Tự xây dựng

### Bước 1: Tất cả các hàm định mức và khoảng cách

Xem `code/distances.py` để biết cách triển khai đầy đủ. Mọi chức năng đều được xây dựng từ đầu chỉ sử dụng toán học Python cơ bản.

### Bước 2: Cùng dữ liệu, khoảng cách khác nhau, hàng xóm khác nhau

Bản demo trong `distances.py` tạo một dataset, chọn một điểm truy vấn và cho thấy hàng xóm gần nhất thay đổi như thế nào tùy thuộc vào số liệu khoảng cách. Điểm "gần nhất" dưới L1 có thể không gần nhất dưới L2 hoặc cosin.

### Bước 3: Embedding tìm kiếm điểm tương đồng

Mã bao gồm một tìm kiếm mô phỏng embedding tương tự tìm kiếm các "tài liệu" giống nhất với một truy vấn bằng cách sử dụng độ tương đồng cosin so với khoảng cách L2, cho thấy rằng thứ hạng có thể khác nhau.

## Ứng dụng

Cách sử dụng thực tế phổ biến nhất: tìm các mục tương tự trong cơ sở dữ liệu vector.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Khi bạn gọi `model.encode(text)` và sau đó tìm kiếm cơ sở dữ liệu vector, đây là những gì xảy ra bên trong. embedding model ánh xạ văn bản đến vectors. Cơ sở dữ liệu vector tính toán sự tương đồng cosin (hoặc tích chấm) giữa vector truy vấn của bạn và mọi vector được lưu trữ, sử dụng thuật toán ANN để tránh kiểm tra tất cả chúng.

## Bài tập

1. Tính toán khoảng cách L1, L2 và L-vô cực giữa (1, 2, 3) và (4, 0, 6). Xác minh rằng L-inf < = L2 < = L1 luôn giữ cho bất kỳ cặp điểm nào. Chứng minh lý do tại sao thứ tự này được đảm bảo.

2. Tạo hai vectors trong đó độ tương đồng cosin cao (> 0,9) nhưng khoảng cách L2 lớn (> 10). Giải thích về mặt hình học những gì đang xảy ra. Sau đó tạo hai vectors trong đó độ tương đồng cosin thấp (< 0,3) nhưng khoảng cách L2 nhỏ (< 0,5).

3. Triển khai một hàm nhận một dataset và một điểm truy vấn và trả về hàng xóm gần nhất theo khoảng cách L1, L2, cosin và Mahalanobis. Tìm một dataset mà cả bốn đều không đồng ý về điểm nào gần nhất.

4. Tính khoảng cách Wasserstein giữa [0,5, 0,5, 0, 0] và [0, 0, 0,5, 0,5] bằng tay bằng phương pháp CDF. Sau đó tính toán nó trong khoảng [0,25, 0,25, 0,25, 0,25] và [0, 0, 0,5, 0,5]. Cái nào lớn hơn và tại sao?

5. Triển khai MinHash để có sự tương đồng gần đúng của Jaccard. Tạo 100 bộ ngẫu nhiên, tính toán Jaccard chính xác cho tất cả các cặp và so sánh với xấp xỉ MinHash bằng cách sử dụng 50, 100 và 200 hàm băm. Vẽ sai số xấp xỉ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Định mức | "Kích thước của một vector" | Một hàm ánh xạ một vector với một vô hướng không âm, thỏa mãn bất đẳng thức tam giác, tính đồng nhất tuyệt đối và chỉ bằng không cho vector |
| Định mức L1 | "Khoảng cách Manhattan" | Tổng các giá trị thành phần tuyệt đối. Tạo ra sự thưa thớt trong tối ưu hóa. Mạnh mẽ đến ngoại lệ |
| Định mức L2 | "Khoảng cách Euclid" | Căn bậc hai của tổng các thành phần bình phương. Khoảng cách đường thẳng trong không gian Euclid |
| Định mức Lp | "Định mức tổng quát" | Căn thứ p của tổng lũy thừa thứ p của các thành phần tuyệt đối. L1 và L2 là những trường hợp đặc biệt |
| Định mức L-vô cực | "Định mức tối đa" hoặc "Khoảng cách Chebyshev" | Giá trị thành phần tuyệt đối tối đa. Giới hạn của Lp khi p tiếp cận vô cực |
| Sự tương đồng cosin | "Góc giữa vectors" | Sản phẩm chấm được chuẩn hóa theo cả hai độ lớn. Phạm vi từ -1 đến +1. Bỏ qua độ dài vector |
| Khoảng cách cosin | "1 trừ sự tương đồng cosin" | Chuyển đổi độ tương tự cosin thành khoảng cách. Phạm vi từ 0 đến 2 |
| Sản phẩm chấm | "Cosin không chuẩn hóa" | Tổng các sản phẩm theo thành phần. Bằng độ tương tự cosin nhân với cả hai độ lớn |
| Khỏang cách Mahalanobis | "Khoảng cách nhận biết tương quan" | Khoảng cách L2 trong một không gian đã được làm trắng (khử tương quan và chuẩn hóa) bằng cách sử dụng ma trận hiệp phương sai dữ liệu |
| Điểm tương đồng của Jaccard | "Đặt chồng chéo" | Kích thước giao lộ chia cho kích thước của công đoàn. Đối với bộ, không vectors |
| Chỉnh sửa khoảng cách | "Khoảng cách Levenshtein" | Chèn, xóa và thay thế tối thiểu để chuyển đổi chuỗi này sang chuỗi khác |
| Phân kỳ KL | "Khoảng cách giữa các phân phối" | Không phải là khoảng cách thực (không đối xứng). Đo các bit bổ sung từ việc sử dụng Q để mã hóa P |
| Khỏang cách Wasserstein | "Khoảng cách của người di chuyển trái đất" | Công tối thiểu để transport khối lượng từ phân phối này sang phân phối khác. Một số liệu thực sự |
| Hàng xóm gần nhất gần đúng | "Tìm kiếm ANN" | Các thuật toán (HNSW, LSH, IVF) tìm các điểm gần nhất nhanh hơn nhiều so với tìm kiếm chính xác |
| HNSW | "Thuật toán vector DB" | Biểu đồ Thế giới nhỏ có thể điều hướng phân cấp. Biểu đồ nhiều lớp để tìm kiếm hàng xóm gần nhất nhanh chóng |
| Chính quy hóa L1 | "Dây thừng" | Thêm định mức trọng lượng L1 vào loss. Đẩy trọng lượng về không (thưa thớt) |
| Chính quy hóa L2 | "Sườn núi" hoặc "phân rã trọng lượng" | Cộng định mức bình phương L2 của trọng số vào loss. Thu nhỏ trọng lượng về không mà không thưa thớt |
| Lưới đàn hồi | "L1 + L2" | Kết hợp chính quy hóa L1 và L2. Xử lý các nhóm feature tương quan tốt hơn so với một trong hai |

## Đọc thêm

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Thư viện của Meta dành cho tìm kiếm ANN tỷ quy mô
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - bài báo giới thiệu khoảng cách của Earth Mover với GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - thuật toán ANN cơ bản
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, trong đó sự tương đồng cosin trở thành mặc định cho embeddings
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - hướng dẫn thực hành về các chỉ số khoảng cách và thuật toán lân cận trong scikit-learn
