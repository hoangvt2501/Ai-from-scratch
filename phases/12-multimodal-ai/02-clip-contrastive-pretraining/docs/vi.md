# CLIP và Pretraining ngôn ngữ thị giác tương phản

> CLIP của OpenAI (2021) đã chứng minh một ý tưởng duy nhất đủ lớn để cung cấp năng lượng cho năm năm tới: căn chỉnh encoder hình ảnh và encoder văn bản trong cùng một không gian vector chỉ bằng cách sử dụng các cặp chú thích hình ảnh web nhiễu và loss tương phản. Không có nhãn được giám sát. 400 triệu cặp. Không gian embedding kết quả zero-shot phân loại, truy xuất văn bản hình ảnh và cắm vào mỗi VLM 2026 làm tháp tầm nhìn của nó. SigLIP 2 (2025) đã thay thế softmax bằng sigmoid và mở rộng quy mô qua CLIP với chi phí thấp hơn. Bài học này đi bộ toán học từ InfoNCE sang sigmoid theo cặp loss và xây dựng bước training trong Python stdlib.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Kiến thức tiên quyết:** Giai đoạn 12 · 01 (Bản vá ViT), Giai đoạn 7 (Transformers)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Lấy thông tin NCE loss từ thông tin lẫn nhau và triển khai phiên bản vector hóa ổn định về số lượng.
- Giải thích lý do tại sao loss theo cặp sigmoid (SigLIP) mở rộng quy mô thành batch 32768+ mà không cần yêu cầu chi phí softmax tập hợp tất cả.
- Chạy zero-shot phân loại ImageNet bằng cách xây dựng các mẫu văn bản (`a photo of a {class}`) và lấy argmax thay vì sự tương đồng cosin.
- Đặt tên cho bốn đòn bẩy CLIP / SigLIP pretraining cung cấp cho bạn: batch kích thước, temperature, mẫu prompt, chất lượng dữ liệu.

## Vấn đề

Tầm nhìn trước CLIP đã được giám sát. Thu thập datasets có nhãn (ImageNet: 1,2 triệu hình ảnh, 1000 classes), huấn luyện CNN ship nó. Nhãn đắt tiền, nhãn bias những gì người dán nhãn có thể đồng ý và nhãn không chuyển sang các nhiệm vụ mới mà không tinh chỉnh.

Web chú thích hình ảnh có hơn một tỷ cặp được dán nhãn lỏng lẻo miễn phí. Một bức ảnh của một săn vàng với văn bản thay thế "Max của tôi trong công viên" mang một tín hiệu giám sát - văn bản mô tả hình ảnh. Câu hỏi: bạn có thể biến điều này thành training hữu ích không?

Câu trả lời của CLIP: coi các cặp chú thích hình ảnh như một nhiệm vụ phù hợp. Với một batch gồm N hình ảnh và N chú thích, hãy học cách khớp từng hình ảnh với chú thích riêng của nó để chống lại những người gây xao nhãng N-1. Sự giám sát là "hai thứ này thuộc về nhau; những N-1 này thì không." Không có nhãn class. Không có chú thích của con người. Chỉ là một loss tương phản.

Kết quả là không gian embedding làm được nhiều hơn những gì CLIP đã được huấn luyện. ImageNet zero-shot hoạt động vì "ảnh mèo" nhúng gần hình ảnh của những con mèo chưa bao giờ được dán nhãn rõ ràng là mèo. Đây là cược xuất hiện vào mỗi VLM năm 2026.

## Khái niệm

### encoder kép

CLIP có hai tháp:

- encoder `f` hình ảnh: ViT hoặc ResNet, xuất ra vector D-dim cho mỗi hình ảnh.
- encoder `g` văn bản: transformer nhỏ, xuất ra vector D-dim cho mỗi chú thích.

Cả hai tháp đều chuẩn hóa đầu ra của chúng thành chiều dài đơn vị. Sự tương đồng là `cos(f(x), g(y)) = f(x)^T g(y)` vì cả hai đều là định mức đơn vị.

Đối với batch các cặp N (hình ảnh, chú thích), hãy xây dựng ma trận tương tự `S` hình dạng `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

trong đó `tau` là một temperature đã học (CLIP khởi tạo thành 0,07; đã học trong log-space).

### Thông tin NCE loss

CLIP sử dụng entropy chéo đối xứng trên các hàng và cột:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Đây là InfoNCE. softmax trong CE buộc mỗi hình ảnh phải khớp với chú thích của nó nhiều hơn mọi chú thích khác trong batch. "Âm bản" là tất cả các mục batch khác. batches lớn hơn = nhiều âm bản hơn = tín hiệu mạnh hơn. CLIP được huấn luyện ở batch 32k; tỷ lệ quan trọng.

### Temperature

`tau` kiểm soát độ sắc nét của softmax. Tau thấp → phân bố sắc nét, hiệu ứng khai thác âm cứng. Tau cao → mềm, tất cả các samples đóng góp. CLIP học log (1/tau), được cắt để tránh sụp đổ. SigLIP 2 sửa tau ban đầu và thay vào đó sử dụng một bias đã học.

### Tại sao thang đo sigmoid tốt hơn (SigLIP)

Softmax cần toàn bộ ma trận tương tự đồng bộ. Trong training phân tán, bạn phải tập hợp tất cả mọi embedding đến mọi bản sao, sau đó thực hiện softmax. Đây là bậc hai về kích thước thế giới để giao tiếp.

SigLIP thay thế softmax bằng sigmoid theo nguyên tố: đối với mỗi cặp `(i, j)`, loss là một phân loại nhị phân của "đây có phải là cặp phù hợp không?" dương class nhãn là đường chéo, mọi thứ khác là âm. loss là:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` nếu `i == j`, nếu không là 0. loss của mỗi cặp là độc lập. Không cần tất cả thu thập. Mỗi GPU tính toán khối cục bộ và tổng của nó. SigLIP 2 mở rộng tỷ lệ batch 32k-512k với giá rẻ trong khi CLIP sẽ cần giao tiếp nhiều hơn theo tỷ lệ.

### Phân loại Zero-shot

Cho N tên class, đối với mỗi class xây dựng một mẫu văn bản:

```
"a photo of a {class}"
```

Nhúng mỗi mẫu với encoder văn bản. Nhúng hình ảnh của bạn với encoder hình ảnh. Độ tương đồng cosin Argmax = class dự đoán. Không có training trên classes đích.

Prompt mẫu quan trọng. Giấy gốc của CLIP sử dụng 80 mẫu mỗi class (trơn, nghệ thuật, ảnh, tranh, v.v.) và tính trung bình embeddings. +3 điểm ImageNet. Cách sử dụng hiện đại thường chọn một hoặc hai mẫu.

### Đầu dò tuyến tính và tinh chỉnh

Zero-shot là đường cơ sở. Một đầu dò tuyến tính (huấn luyện một lớp tuyến tính trên features CLIP bị đóng băng cho classes mục tiêu của bạn) đánh bại zero-shot trong các tác vụ trong miền. Tinh chỉnh hoàn toàn đánh bại đầu dò tuyến tính trên trong miền nhưng có thể ảnh hưởng đến quá trình chuyển zero-shot. Ba chế độ với ba sự đánh đổi.

### SigLIP 2: NaFlex và features dày đặc

SigLIP 2 (2025) bổ sung:
- NaFlex: model đơn xử lý tỷ lệ khung hình và độ phân giải thay đổi.
- features mật độ tốt hơn để phân đoạn và ước tính độ sâu, nhắm mục tiêu sử dụng như một xương sống đông lạnh trong VLMs.
- Đa ngôn ngữ: được huấn luyện trên 100+ ngôn ngữ trong đó CLIP chỉ có tiếng Anh.
- Thang đo tham số 1B trong đó CLIP đạt đỉnh ở 400M.

Vào năm 2026 mở VLMs, SigLIP 2 SO400m/14 là tháp tầm nhìn mặc định. CLIP vẫn là mặc định để truy xuất văn bản hình ảnh thuần túy trong đó phân phối training LAION-2B cụ thể khớp với mẫu truy vấn của bạn.

### CĂN CHỈNH, CƠ BẢN, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): ý tưởng tương tự như CLIP, tỷ lệ cặp 1.8B, 90% nhiễu. Quy mô dữ liệu nhiễu đã được chứng minh. OpenCLIP (LAION): tái tạo mở CLIP trên LAION-400M / 2B, nhiều tỷ lệ, checkpoint mở. EVA-CLIP: khởi tạo từ mô hình hình ảnh được che giấu; xương sống mạnh mẽ cho VLMs. BASIC: Kết hợp CLIP + ALIGN của Google. Tất cả cùng một họ, dữ liệu và điều chỉnh khác nhau.

### Trần zero-shot

Giới hạn CLIP-class models khoảng 76% zero-shot ImageNet (CLIP-G, OpenCLIP-G). Beyond yêu cầu dữ liệu lớn hơn nhiều (SigLIP 2 nhận được 80% +) hoặc thay đổi kiến trúc (đầu được giám sát, parameters hơn). benchmark đang bão hòa; giá trị thực là không gian embedding mà hạ lưu VLMs tiêu thụ.

```figure
multimodal-fusion
```

## Ứng dụng

`code/main.py` thực hiện:

1. Một encoder kép đồ chơi (features hình ảnh dựa trên băm, ký tự văn bản features) để bạn có thể xem hình dạng InfoNCE mà không cần numpy.
2. InfoNCE loss trong Python thuần túy (ổn định số thông qua log-sum-exp).
3. Sigmoid theo cặp loss để so sánh.
4. Một quy trình phân loại zero-shot: tính toán sự tương đồng cosin so với một tập hợp các prompts văn bản, argmax để dự đoán.

Chạy nó và xem đường cong loss. Các con số tuyệt đối là đồ chơi; hình dạng phù hợp với những gì một huấn luyện viên CLIP thực sự phát ra.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-clip-zero-shot.md`. Với một tập hợp hình ảnh (thông qua đường dẫn) và danh sách các classes mục tiêu, nó xây dựng prompts văn bản bằng mẫu CLIP, nhúng cả hai bên với một checkpoint đã nêu (ví dụ: `openai/clip-vit-large-patch14`) và trả về các dự đoán top 1 / top 5 với điểm tương tự. skill từ chối đưa ra tuyên bố về classes không có trong danh sách prompt.

## Bài tập

1. Triển khai InfoNCE cho batch 4 cặp bằng tay. Xây dựng ma trận tương tự 4x4, chạy softmax, chọn đường chéo, tính entropy chéo. Xác minh việc triển khai Python của bạn dựa trên phép tính ván tay này.

2. SigLIP sử dụng một bias parameter `b` ngoài temperature: `S'[i,j] = S[i,j]/tau + b`. `b` đóng vai trò gì khi batch có class imbalance lớn (nhiều âm tính hơn dương tính trên mỗi hàng)? Đọc SigLIP Phần 3 (arXiv: 2303.15343).

3. Xây dựng một bộ phân loại zero-shot cho mèo và chó. Hãy thử hai mẫu prompt: `a photo of a {class}` và `a picture of a {class}`. Đo accuracy trên 100 hình ảnh thử nghiệm. Tập hợp các mẫu có đánh bại đơn không?

4. Tính toán chi phí giao tiếp của softmax InfoNCE so với sigmoid theo cặp để chạy 512-GPU ở batch 32k. Tỷ lệ nào là O (N), tỷ lệ nào là O (N ^ 2)? Trích dẫn SigLIP Phần 4.

5. Đọc bài báo về quy tắc tỷ lệ OpenCLIP (arXiv:2212.07143, Cherti et al.). Tái tạo kết luận của họ để chia tỷ lệ dữ liệu từ các hình: ở kích thước model cố định, mối quan hệ log-tuyến tính giữa ImageNet zero-shot accuracy và kích thước dữ liệu training là gì?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Thông tin NCE | "loss tương phản" | Entropy chéo trên ma trận tương tự của một batch; dương của mỗi mục là mục được ghép nối của nó, âm bản là mọi thứ khác |
| Sigmoid loss | "SigLIP loss" | Entropy chéo nhị phân trên mỗi cặp; không có softmax, không tập hợp tất cả, quy mô rẻ trong training phân tán |
| Temperature | "Tàu" | Vô hướng chia tỷ lệ logits trước khi softmax/sigmoid; kiểm soát độ sắc nét của phân phối |
| Zero-shot | "Phân loại không tinh chỉnh" | Sử dụng prompts văn bản để xây dựng class embeddings và phân loại theo độ tương tự cosin; không có training trên classes mục tiêu |
| Mẫu Prompt | "Một bức ảnh của..." | Giàn giáo văn bản xung quanh tên class; ảnh hưởng đến zero-shot accuracy 1-5 điểm |
| encoder kép | "Hai tòa tháp" | Một encoder hình ảnh + một encoder văn bản, đầu ra trong không gian D-dim dùng chung |
| Tiêu cực cứng | "Kẻ gây xao nhãng khó khăn" | Một tiêu cực đủ tương tự với tích cực đến mức model phải làm việc để tách chúng ra |
| Đầu dò tuyến tính | "Đông lạnh + một lớp" | Chỉ huấn luyện một bộ phân loại tuyến tính trên features đông lạnh; đo lường chất lượng feature |
| NaFlex | "Độ phân giải linh hoạt gốc" | Khả năng nhập hình ảnh ở bất kỳ tỷ lệ khung hình và độ phân giải nào mà không cần thay đổi kích thước |
| Temperature mở rộng quy mô | "tau tham số log" | CLIP tham số hóa `log(1/tau)` để gradients hoạt động; clip để ngăn chặn sự sụp đổ xuống tau gần bằng không |

## Đọc thêm

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) - giấy CLIP.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) - SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) - đa ngôn ngữ + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) - mở rộng quy mô với dữ liệu web nhiễu.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) - Luật tỷ lệ OpenCLIP.
