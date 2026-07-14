# Tắc kè hoa và Models đa phương thức chỉ Token hợp nhất sớm

> Mọi VLM chúng ta đã thấy cho đến nay đều giữ hình ảnh và văn bản riêng biệt. tokens hình ảnh đến từ một tầm nhìn encoder, chảy vào máy chiếu, sau đó gặp văn bản bên trong LLM. Tầm nhìn và từ vựng văn bản không bao giờ trùng lặp. Chameleon (Meta, tháng 5 năm 2024) hỏi: điều gì sẽ xảy ra nếu họ làm vậy? Huấn luyện VQ-VAE biến hình ảnh thành một chuỗi các tokens rời rạc từ một từ vựng được chia sẻ. Mỗi tài liệu đa phương thức bây giờ là một chuỗi - tokens văn bản và hình ảnh tokens xen kẽ, một loss tự hồi quy duy nhất. Tác dụng phụ: model có thể tạo ra các đầu ra phương thức hỗn hợp - xen kẽ tokens văn bản và hình ảnh trong một cuộc gọi inference duy nhất. Bài học này đọc luận điểm hợp nhất ban đầu và xây dựng một phiên bản đồ chơi từ đầu đến cuối.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05, Giai đoạn 8 (Generative AI)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích lý do tại sao từ vựng được chia sẻ + loss đơn lẻ thay đổi những gì model có thể làm.
- Mô tả cách VQ-VAE mã hóa hình ảnh thành một chuỗi rời rạc tương thích với mục tiêu token tiếp theo của transformer.
- Kể tên các thủ thuật ổn định training của Tắc kè hoa: QK-Norm, vị trí dropout LayerNorm thứ tự.
- So sánh cách tiếp cận Q-Former của Chameleon và BLIP-2 và mô tả khi nào mỗi lựa chọn là lựa chọn phù hợp.

## Vấn đề

VLMs dựa trên bộ điều hợp (LLaVA, BLIP-2, Qwen-VL) coi văn bản và hình ảnh là hai thứ khác nhau. Một văn bản token đi qua `embed(text_token)`; một hình ảnh đi qua `visual_encoder(image) → projector → ... pseudo_tokens`. model có hai đường dẫn đầu vào merge đi vào một nửa.

Ba hậu quả:

1. Người LLM chỉ có thể tiêu thụ hình ảnh chứ không thể phát ra chúng. Đầu ra chỉ là văn bản.
2. Các tài liệu phương thức hỗn hợp (xen kẽ các đoạn văn và hình ảnh, như trong một bài báo) rất khó xử - bạn phân tích cú pháp đầu vào đa phương thức bên ngoài thế hệ model hoặc chuỗi.
3. Phân phối không khớp. tokens hình ảnh và văn bản tokens sống ở các vùng khác nhau của không gian ẩn, tạo ra các vấn đề alignment tinh tế.

Chameleon bác bỏ tiền đề: hình ảnh chỉ là chuỗi các tokens rời rạc từ một từ vựng được chia sẻ. Huấn luyện model trên các tài liệu xen kẽ, một loss, một decoder tự hồi quy và bạn mở khóa tạo phương thức hỗn hợp miễn phí.

## Khái niệm

### VQ-VAE dưới dạng hình ảnh tokenizer

tokenizer là một bộ mã hóa tự động biến thiên lượng tử hóa vector. Kiến trúc:

- Encoder: CNN + ViT ánh xạ hình ảnh với bản đồ feature không gian, giả sử 32x32 features của mờ 256.
- Sách mật mã: một từ vựng đã học của K vectors (Tắc kè hoa sử dụng 8192), cũng mờ 256.
- Quantization: đối với mỗi feature không gian, tra cứu mục nhập codebook gần nhất theo khoảng cách L2. Thay thế feature liên tục bằng chỉ số nguyên.
- Decoder: CNN đưa features lượng tử hóa trở lại pixel.

Training: loss tái thiết VAE + loss cam kết + loss codebook. Các chỉ mục sách mã tạo thành một bảng chữ cái rời rạc cho hình ảnh.

Đối với tắc kè hoa: một hình ảnh trở thành 32 * 32 = 1024 tokens được rút ra từ từ vựng 8192. Nối với tokens văn bản (từ từ vựng BPE của LLM, ví dụ 32000). Từ vựng cuối cùng: 40192. transformer thấy một chuỗi, một loss.

### Từ vựng được chia sẻ

Từ vựng của Chameleon kết hợp tokens văn bản, tokens hình ảnh và dấu phân cách phương thức. Mỗi token có một ID duy nhất. Lớp embedding đầu vào ánh xạ mọi ID với vector ẩn D-dim. Phép chiếu đầu ra ẩn trở lại logits từ vựng. Softmax chọn token tiếp theo, bất kể phương thức nào.

Dấu phân cách quan trọng: Thẻ `<image>` và `</image>` bao quanh chuỗi token hình ảnh. Tại thời điểm tạo ra, nếu model phát ra `<image>`, phần mềm xuôi dòng biết 1024 tokens tiếp theo là chỉ số VQ để gửi đến decoder để hiển thị pixel.

### Tạo phương thức hỗn hợp

Inference là dự đoán token tiếp theo trong từ vựng được chia sẻ. Ví dụ prompt: "Vẽ một con mèo và mô tả nó." Tắc kè hoa phát ra:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

model chọn đơn đặt hàng một cách tự động - nó có thể tạo ra hình ảnh sau đó là văn bản, văn bản sau đó là hình ảnh hoặc xen kẽ. Cùng decoder, cùng loss.

So sánh với bộ chuyển đổi VLMs nơi việc tạo chỉ có văn bản. Chameleon mở lại câu hỏi về phương thức đầu ra model.

### Training ổn định - QK-Norm, dropout LayerNorm đặt hàng

training hợp nhất sớm không ổn định ở quy mô lớn. Bài báo của Chameleon ghi lại ba thủ thuật:

- QK-Định mức. Áp dụng LayerNorm cho truy vấn và phép chiếu khóa bên trong attention, trước sản phẩm chấm. Ngăn chặn vụ nổ cường độ logit ở độ sâu. Được sử dụng bởi nhiều models lớn sau năm 2024.
- Dropout vị trí. Dropout sau mỗi lần cộng dư, không chỉ sau attention và MLP. Yêu cầu chính quy hóa hơn khi gradients từ hình ảnh tokens có thể chiếm ưu thế.
- LayerNorm đặt hàng. Pre-LN trên branch dư (tiêu chuẩn), cộng thêm một LN trên kết nối bỏ qua của khối cuối cùng. Ổn định dòng gradient lớp cuối cùng.

Nếu không có những thủ thuật này, Chameleon 34-param training phân kỳ ở nhiều checkpoints. Với họ, nó hội tụ. Công thức training cũng đóng góp nhiều như kiến trúc.

### Trần tái thiết tokenizer

VQ-VAE bị mất dữ liệu. Với 8192 mục nhập sách mã và 1024 tokens trên mỗi hình ảnh 512x512, PSNR tái tạo giới hạn khoảng 26-28 dB. Điều này là đủ để tạo hình ảnh dễ nhận biết nhưng kém hơn rõ ràng so với khuếch tán không gian liên tục (Khuếch tán ổn định 3 đạt 32+ dB).

tokenizer là nút thắt cổ chai. Tốt hơn tokenizers (MAGVIT-v2, IBQ, SBER-MoVQGAN) nâng trần nhà. Emu3 (Bài 12.12) đạt được chất lượng SDXL chỉ thông qua tokenizer tốt hơn.

### Tắc kè hoa so với BLIP-2 / LLaVA

Tắc kè hoa (hợp nhất sớm, từ vựng chung):
- Một loss, một decoder.
- Tạo ra đầu ra phương thức hỗn hợp.
- Tokenizer là trần chất lượng.
- Đắt: VQ-VAE decoder cho mỗi hình ảnh được tạo trên đường dẫn inference.

BLIP-2 / LLaVA (nhiệt hạch muộn, tháp riêng biệt):
- Tầm nhìn vào, chỉ văn bản ra.
- Tái sử dụng pretrained LLM.
- Không tokenizer nút thắt cổ chai cho sự hiểu biết.
- Giá rẻ: forward pass đơn.

Chọn theo nhiệm vụ. Nếu bạn cần tạo hình ảnh, gia đình tắc kè hoa. Nếu bạn chỉ cần hiểu biết, adapter-VLM đơn giản hơn và sử dụng lại nhiều pretrained tính toán hơn.

### Fuyu và AnyGPT

Fuyu (Adept, 2023) là một cách tiếp cận liên quan: bỏ qua hoàn toàn encoder tầm nhìn riêng biệt, cung cấp các bản vá hình ảnh thô thông qua phép chiếu đầu vào của LLM như thể chúng tokens, không tokenizer. Đơn giản hơn Chameleon, mất đi việc tạo đầu ra từ vựng được chia sẻ.

AnyGPT (Zhan và cộng sự, 2024) mở rộng Tắc kè hoa thành bốn phương thức: văn bản, hình ảnh, lời nói, âm nhạc. Cùng một thủ thuật VQ-VAE cho mỗi người, chia sẻ transformer. Bất kỳ thế hệ nào. Được đề cập nhiều hơn trong Bài 12.16.

## Ứng dụng

`code/main.py` xây dựng một model nhiệt hạch sớm từ đầu đến cuối:

- Một bộ định lượng kiểu VQ-VAE nhỏ ánh xạ các bản vá 8x8 với các chỉ mục sách mã (K = 16).
- Từ vựng được chia sẻ là (id văn bản 0..31) + (id hình ảnh 32..47) + (dấu phân cách 48, 49).
- Một decoder tự hồi quy đồ chơi (bảng bigram) được huấn luyện về chú thích tổng hợp + chuỗi token hình ảnh.
- Sampling vòng lặp phát ra văn bản + hình ảnh xen kẽ tokens cho một prompt.

Mã cố tình giữ cho các transformer nhỏ (bigram) để bạn có thể trace luồng tín hiệu từ đầu đến cuối.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-tokenizer-vs-adapter-picker.md`. Với thông số kỹ thuật sản phẩm (chỉ hiểu so với hiểu + tạo, chất lượng hình ảnh yêu cầu, ngân sách chi phí), nó chọn giữa họ tắc kè hoa (hợp nhất sớm) và họ LLaVA (hợp nhất muộn) và biện minh bằng các quy tắc định lượng.

## Bài tập

1. Chameleon sử dụng các mục nhập sách mã K = 8192 và 1024 tokens trên mỗi hình ảnh 512x512. Ước tính tỷ lệ nén so với hình ảnh RGB 24 bit. Nó có bị mất mát không? Mất mát như thế nào?

2. Một hình ảnh 4K (3840x2160) ở cùng mật độ VQ-VAE tạo ra bao nhiêu hình ảnh tokens? Một model kiểu tắc kè hoa có thể tạo hình ảnh 4K trong một cuộc gọi inference không? Điều gì phá vỡ đầu tiên - ngữ cảnh, chất lượng tokenizer hay KV cache?

3. Triển khai QK-Norm trong Python thuần túy. Cho truy vấn và khóa 64 mờ, hãy hiển thị sản phẩm chấm trước và sau LayerNorm. Tại sao kiểm soát cường độ lại quan trọng ở độ sâu?

4. Đọc Chameleon Phần 2.3 về độ ổn định training. Mô tả chế độ lỗi chính xác mà bài báo quan sát được ở 34B mà không có QK-Norm. Chữ ký "bùng nổ chuẩn mực" là gì?

5. Mở rộng decoder đồ chơi để phát ra phản hồi phương thức hỗn hợp với prompt chỉ có văn bản. Đo lường tần suất model chọn hình ảnh đầu tiên so với văn bản trước khi phân phối dữ liệu training 60% văn bản trước / 40% hình ảnh trước.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Hợp nhất sớm | "tokens thống nhất" | Hình ảnh được chuyển đổi thành tokens rời rạc chia sẻ từ vựng của transformer từ bước một |
| VQ-VAE | "Hình ảnh tokenizer" | Sách mã CNN + ViT + ánh xạ hình ảnh thành các chỉ số số nguyên mà transformer có thể dự đoán |
| Từ vựng được chia sẻ | "Một từ điển" | Một khoảng trống token ID duy nhất bao gồm các dấu phân cách văn bản + hình ảnh + phương thức |
| QK-Định mức | "Attention ổn định" | LayerNorm áp dụng cho truy vấn và khóa trước sản phẩm chấm của họ, ngăn chặn thổi phồng định mức |
| Tạo phương thức hỗn hợp | "Đầu ra văn bản + hình ảnh" | Inference tự động tạo tokens văn bản và hình ảnh xen kẽ trong một lần |
| Kích thước sách mã | "Mục nhập K" | Số lượng vectors rời rạc mà VQ-VAE có thể lượng tử hóa; Giao dịch nén cho độ trung thực |
| Trần Tokenizer | "Giới hạn tái tạo" | PSNR tốt nhất có thể đạt được bằng cách giải mã VQ tokens; Giới hạn chất lượng hình ảnh của model |

## Đọc thêm

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
