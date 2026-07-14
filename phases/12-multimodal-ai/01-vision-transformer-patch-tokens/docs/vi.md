# Vision Transformers và bản vá Token Primitive

> Trước bất cứ điều gì đa phương thức, một hình ảnh phải trở thành một chuỗi tokens một transformer có thể ăn. Bài báo ViT năm 2020 đã trả lời điều này bằng các bản vá pixel 16x16, phép chiếu tuyến tính và embedding vị trí. Năm năm sau, mỗi model biên giới năm 2026 (Claude Opus 4.7 ở 2576px gốc, Gemini 3.1 Pro, Qwen3.5-Omni) vẫn bắt đầu theo cách này - encoder thay đổi từ ViT sang DINOv2 thành SigLIP 2, thanh ghi tokens được thêm vào, sơ đồ vị trí trở thành 2D-RoPE, nhưng primitive vẫn được giữ nguyên. Bài học này đọc bản vá token pipeline từ đầu đến cuối và xây dựng nó trong stdlib Python để rest của Giai đoạn 12 có một model tinh thần cụ thể cho "tokens thị giác".

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, patch tokenizer + geometry calculator)
**Kiến thức tiên quyết:** Giai đoạn 7 (Transformers), Giai đoạn 4 (Thị giác máy tính)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Chuyển đổi hình ảnh HxWx3 thành một chuỗi tokens bản vá với mã hóa vị trí chính xác.
- Tính toán độ dài trình tự, số lượng parameter và FLOPs cho một ViT nhất định (kích thước bản vá, độ phân giải, độ mờ ẩn, độ sâu).
- Kể tên ba nâng cấp đã đưa ViT từ nghiên cứu năm 2020 đến năm 2026 production: pretraining tự giám sát (DINO / MAE), tokens đăng ký và đóng gói độ phân giải gốc.
- Chọn giữa việc gộp CLS, gộp trung bình và đăng ký tokens cho một nhiệm vụ xuôi dòng.

## Vấn đề

Transformers hoạt động trên các chuỗi vectors. Văn bản đã là một chuỗi (byte hoặc tokens). Hình ảnh là một lưới pixel 2D với ba kênh màu - không phải là một chuỗi. Nếu bạn làm phẳng mỗi pixel, hình ảnh RGB 224x224 sẽ trở thành 150.528 tokens và self-attention ở độ dài đó là không bắt đầu (bậc hai về độ dài chuỗi).

Các phương pháp tiếp cận trước năm 2020 đã bắt vít một bộ trích xuất feature CNN vào mặt trước: ResNet tạo ra một bản đồ feature 7x7 của vectors 2048 mờ, đưa 49 tokens đó vào một transformer. Điều này hoạt động nhưng kế thừa các thành kiến của CNN (phương sai dịch thuật, trường tiếp nhận cục bộ) và mất đi sự thèm muốn của transformer đối với quy mô.

Dosovitskiy et al. (2020) đã đặt câu hỏi thẳng thừng: điều gì sẽ xảy ra nếu chúng ta bỏ qua CNN? Chia hình ảnh thành các bản vá có kích thước cố định (giả sử 16x16 pixel), chiếu tuyến tính từng bản vá thành một vector, thêm embedding vị trí và đưa trình tự vào một transformer vani. Vào thời điểm đó, đây là dị giáo - tầm nhìn không có sự tích chập. Với đủ dữ liệu (JFT-300M, sau đó là LAION), nó đã đánh bại ResNet trên ImageNet và tiếp tục cải thiện.

Đến năm 2026, primitive ViT là nền tảng không thể nghi ngờ. Mỗi tháp tầm nhìn của VLM trọng lượng mở là hậu duệ của một số hậu duệ (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Câu hỏi không còn là "chúng ta có nên sử dụng các bản vá không?" mà là "kích thước bản vá nào, lịch trình giải quyết nào, mục tiêu pretraining gì, mã hóa vị trí nào."

## Khái niệm

### Các bản vá lỗi dưới dạng tokens

Với một `x` hình ảnh của hình dạng `(H, W, 3)` và kích thước bản vá `P`, bạn khắc hình ảnh thành một lưới gồm `(H/P) x (W/P)` các bản vá không chồng chéo. Mỗi bản vá là một khối pixel `P x P x 3`. Làm phẳng mỗi khối lập phương thành một `3 P^2` vector. Áp dụng `W_E` chiếu tuyến tính được chia sẻ của hình dạng `(3 P^2, D)` ánh xạ mỗi bản vá vào `D` kích thước ẩn của model.

Đối với config chuẩn ViT-B/16:
- Độ phân giải 224, kích thước bản vá 16 → lưới 14x14 → 196 bản vá tokens.
- Mỗi bản vá là `16 x 16 x 3 = 768` giá trị pixel, được chiếu đến `D = 768`.
- Thêm độ dài chuỗi `[CLS]` token → có thể học được 197.

Phép chiếu bản vá giống hệt về mặt toán học với tích chập 2D với kích thước hạt nhân `P`, `P` sải chân và `D` các kênh đầu ra. Đó là cách mã production thực sự triển khai nó - `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Khung "phép chiếu tuyến tính" là khái niệm; khung hạt nhân hiệu quả.

### embeddings vị trí

Các bản vá không có thứ tự cố hữu - transformer coi chúng như một cái túi. ViT ban đầu đã thêm một embedding vị trí 1D có thể học được (một vector 768 mờ cho mỗi vị trí, 197 trong số đó). Hoạt động, nhưng gắn model với độ phân giải training: tại inference bạn phải nội suy bảng vị trí nếu bạn thay đổi lưới.

Đường trục thị giác hiện đại sử dụng 2D-RoPE (M-RoPE của Qwen2-VL, mặc định của SigLIP 2) hoặc vị trí 2D được phân tích. 2D-RoPE xoay truy vấn và vectors khóa dựa trên chỉ mục (hàng, cột) của bản vá, vì vậy model suy ra vị trí 2D tương đối từ góc quay. Không có bảng vị trí. model xử lý kích thước lưới tùy ý ở inference.

### CLS token, đầu ra gộp và tokens đăng ký

Biểu diễn cấp hình ảnh là gì? Ba lựa chọn cùng tồn tại:

1. `[CLS]` token. Thêm một vector có thể học được vào chuỗi bản vá. Sau tất cả các khối transformer, trạng thái ẩn của CLS token là biểu diễn hình ảnh. Kế thừa từ BERT. Được sử dụng bởi ViT, CLIP gốc.
2. Hồ bơi trung bình. Tính trung bình các trạng thái ẩn đầu ra tokens bản vá. Được sử dụng bởi SigLIP, DINOv2, VLMs hiện đại nhất.
3. Đăng ký tokens. Darcet et al. (2023) quan sát thấy rằng ViT được huấn luyện mà không có chìm rõ ràng token phát triển các bản vá "artifact" định mức cao chiếm quyền điều khiển self-attention. Thêm 4–16 thanh ghi có thể học được tokens sẽ hấp thụ tải trọng này và cải thiện chất lượng dự đoán mật độ (phân đoạn, độ sâu). DINOv2 và SigLIP 2 đều ship với thanh ghi.

Sự lựa chọn quan trọng đối với các tác vụ xuôi dòng. CLS tốt để phân loại. Đối với VLMs bản vá nguồn cấp dữ liệu đó tokens thành một LLM, bạn bỏ qua hoàn toàn việc gộp chung — mỗi bản vá trở thành một token đầu vào LLM. Các thanh ghi bị loại bỏ trước khi chuyển giao (chúng là giàn giáo, không phải nội dung).

### Pretraining: giám sát, tương phản, che nắng, tự chưng cất

ViT 2020 đã được pretrained với phân loại giám sát trên JFT-300M. Nhanh chóng được thay thế bởi:

- CLIP (2021): hình ảnh-văn bản tương phản trên 400 triệu cặp. Bài 12.02.
- MAE (2021, He et al.): che 75% bản vá, tái tạo pixel. Tự giám sát, hoạt động trên hình ảnh thuần túy.
- DINO (2021) / DINOv2 (2023): tự distillation với học sinh-giáo viên, không nhãn, không chú thích. ViT-g/14 DINOv2023 2 là xương sống trực quan thuần túy mạnh nhất và là mặc định cho các trường hợp sử dụng "features dày đặc".
- SigLIP / SigLIP 2 (2023, 2025): CLIP với loss sigmoid và NaFlex cho tỷ lệ khung hình gốc. Tháp tầm nhìn thống trị vào năm 2026 mở VLMs (Qwen, Idefics2, LLaVA-OneVision).

Lựa chọn pretraining của bạn xác định đường trục tốt cho điều gì: CLIP/SigLIP để khớp ngữ nghĩa với văn bản, DINOv2 cho features hình ảnh dày đặc, MAE làm điểm khởi đầu để tinh chỉnh xuôi dòng.

### Luật tỷ lệ

Mở rộng quy mô ViT (Zhai et al. 2022) đã xác định rằng chất lượng của ViT tuân theo các quy luật có thể dự đoán được về kích thước model, kích thước dữ liệu và tính toán. Ở điện toán cố định:
- model lớn hơn + nhiều dữ liệu hơn → chất lượng tốt hơn.
- Kích thước bản vá là một đòn bẩy về độ dài trình tự so với độ trung thực. Bản vá 14 (điển hình cho DINOv2/SigLIP SO400m) cho nhiều tokens hơn trên mỗi hình ảnh so với bản vá 16; tốt hơn cho các tác vụ OCR và dày đặc, kém hơn về tốc độ.
- Độ phân giải là đòn bẩy lớn khác. Đi từ 224 đến 384 đến 512 hầu như luôn hữu ích, với chi phí bậc hai trong FLOPs.

ViT-g/14 (tham số 1B, bản vá 14, độ phân giải 224 → 256 tokens) và SigLIP SO400m/14 (tham số 400 triệu, bản vá 14) là hai encoders cho VLMs mở năm 2026.

### Parameter tính cho một ViT

Tính toán đầy đủ nằm trong `code/main.py`. Đối với ViT-B/16 tại 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Đỗ bóng mọi ViT theo cách này trước khi bạn tải checkpoint. Kích thước xương sống đặt sàn VRAM của bạn ở bất kỳ VLM hạ lưu nào.

### production config năm 2026

encoder VLMs ship mở nhất vào năm 2026 là SigLIP 2 SO400m/14 ở độ phân giải gốc (NaFlex). Nó có:
- 400 triệu parameters.
- Kích thước bản vá 14, độ phân giải mặc định 384 → 729 tokens bản vá cho mỗi hình ảnh.
- Nhóm trung bình cho các tác vụ cấp hình ảnh; tất cả 729 bản vá đi vào LLM cho VQA.
- 4 đăng ký tokens, loại bỏ trước khi bàn giao LLM.
- 2D-RoPE với tỷ lệ mức hình ảnh cho tỷ lệ khung hình gốc.

Mọi quyết định trong đó đều config traces trở lại một bài báo mà bạn có thể đọc.

```figure
image-patch-tokens
```

## Ứng dụng

`code/main.py` là một bản vá tokenizer và máy tính hình học. Nó lấy (hình ảnh H, W, bản vá P, D ẩn, độ sâu L) và báo cáo:

- Hình dạng lưới và độ dài trình tự sau khi vá.
- Token trình tự cho hình ảnh đồ chơi 8x8 pixel tổng hợp (đi qua đường dẫn Flatten + Project).
- Parameter đếm được chia nhỏ theo nhúng bản vá, nhúng vị trí, khối transformer và đầu.
- FLOPs mỗi forward pass ở độ phân giải mục tiêu.
- Bảng so sánh trên ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Chạy nó. Khớp số lượng parameter với các số đã công bố. Chơi với kích thước và độ phân giải của bản vá để cảm nhận chi phí đếm token.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-patch-geometry-reader.md`. Với config ViT (kích thước bản vá, độ phân giải, độ mờ ẩn, độ sâu), nó tạo ra số lượng token, số lượng parameter và ước tính VRAM với các lý do. Sử dụng skill này bất cứ khi nào bạn chọn xương sống tầm nhìn cho một VLM - nó ngăn chặn những bất ngờ "tokens phát nổ và bối cảnh LLM của tôi được lấp đầy".

## Bài tập

1. Tính toán độ dài trình tự token bản vá cho Qwen2.5-VL ở đầu vào 1280x720 gốc với kích thước bản vá 14. Làm thế nào để so sánh với biểu diễn chỉ CLS?

2. Khung hình 1080p (1920x1080) ở bản vá 14 tạo ra bao nhiêu tokens? Ở tốc độ 30 FPS trong video dài 5 phút, tổng số tokens hình ảnh là bao nhiêu? Chi phí nào giúp bạn tiết kiệm nhất: gộp, sampling khung hình hay hợp nhất token?

3. Thực hiện gộp trung bình trên bản vá tokens trong Python thuần túy. Xác minh rằng nhóm trung bình trên 196 tokens đầu ra DINOv2 khớp với `forward` của model trả về khi bạn yêu cầu embedding gộp.

4. Đọc Phần 3 của "Tầm nhìn Transformers cần thanh ghi" (arXiv: 2309.16588). Mô tả trong hai câu những gì artifact các thanh ghi hấp thụ và tại sao nó lại quan trọng đối với dự đoán dày đặc hạ lưu.

5. Sửa đổi `code/main.py` để hỗ trợ patch-n'-pack: đưa ra một danh sách các hình ảnh có độ phân giải khác nhau, tạo ra một trình tự đóng gói duy nhất và mặt nạ attention đường chéo khối. Xác minh so với Bài 12.06 khi bạn đạt đến nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Bản vá | "Hình vuông 16x16 pixel" | Một vùng không chồng chéo có kích thước cố định của hình ảnh đầu vào; trở thành một token |
| Bản vá embedding | "Phép chiếu tuyến tính" | Ma trận đã học được chia sẻ (hoặc Conv2d với sải chân = P) ánh xạ các pixel bản vá phẳng thành vectors D-mờ |
| CLS token | "Class token" | Các vector có thể học được được thêm vào có trạng thái ẩn cuối cùng đại diện cho toàn bộ hình ảnh; tùy chọn vào năm 2026 |
| Đăng ký token | "Chìm token" | Các tokens có thể học thêm hấp thụ các attention artifacts ViT tiêu chuẩn cao phát triển trong quá trình pretraining |
| Vị trí embedding | "Thông tin vị trí" | Mỗi vị trí vector hoặc xoay làm cho trình tự nhận biết; 2D-RoPE là mặc định hiện đại |
| Lưới | "Lưới vá" | Mảng bản vá (H/P) x (W/P) 2D cho một độ phân giải và kích thước bản vá nhất định |
| NaFlex | "Độ phân giải linh hoạt gốc" | SigLIP 2 feature: model đơn phục vụ nhiều tỷ lệ khung hình và độ phân giải mà không cần huấn luyện lại |
| Xương sống | "Tháp tầm nhìn" | Hình ảnh pretrained encoder có đầu ra token bản vá cung cấp LLM trong một VLM |
| Gộp chung | "Tóm tắt cấp độ hình ảnh" | Chiến lược biến bản vá tokens thành một vector: CLS, mean, attention pool hoặc register-based |
| Bản vá 14 so với 16 | "Lưới mịn hơn và thô hơn" | Bản vá 14 tạo ra nhiều tokens hơn cho mỗi hình ảnh, độ trung thực tốt hơn cho OCR, chậm hơn; Bản vá 16 là mặc định cổ điển |

## Đọc thêm

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT gốc.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) - MAE, pretraining tự giám sát.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) - tự distillation trên quy mô lớn, không có nhãn.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) - đăng ký tokens và phân tích artifact.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) - tháp tầm nhìn mặc định năm 2026.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) - luật tỷ lệ thực nghiệm.
