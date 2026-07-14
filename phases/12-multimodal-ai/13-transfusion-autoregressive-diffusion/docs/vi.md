# Truyền máu: Văn bản tự hồi quy + Hình ảnh khuếch tán trong một Transformer

> Chameleon và Emu3 đặt cược mọi thứ vào tokens rời rạc. Chúng hoạt động, nhưng nút thắt cổ chai quantization có thể nhìn thấy - chất lượng hình ảnh ổn định dưới models khuếch tán không gian liên tục. Transfusion (Meta, Zhou et al., Tháng 8 năm 2024) đặt cược ngược lại: giữ hình ảnh liên tục, bỏ hoàn toàn VQ-VAE và huấn luyện một transformer với hai lần thua. Nhắn tin tokens nhận được dự đoán token tiếp theo. Các bản vá hình ảnh nhận được loss phù hợp / khuếch tán luồng. Cả hai mục tiêu đều tối ưu hóa cùng một trọng số. Kiến trúc cơ bản của Stable Diffusion 3 (MMDiT) là một người anh em họ gần gũi. Bài học này đọc luận án Truyền máu, chế tạo một huấn luyện viên hai loss đồ chơi và traces mặt nạ attention cho phép một người transformer làm cả hai công việc.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Kiến thức tiên quyết:** Giai đoạn 12 · 11 (Tắc kè hoa), Giai đoạn 8 (Generative AI)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Nối một transformer chạy hai tổn thất (NTP trên tokens văn bản, MSE khuếch tán trên các bản vá hình ảnh) trên một đường trục.
- Giải thích lý do tại sao attention hai chiều trên các bản vá hình ảnh cộng với attention nhân quả trên tokens văn bản là lựa chọn mặt nạ phù hợp.
- So sánh kiểu truyền máu (hình ảnh liên tục, loss khuếch tán) với kiểu tắc kè hoa (hình ảnh rời rạc, NTP) về tính toán, chất lượng và độ phức tạp của mã.
- Đặt tên đóng góp của MMDiT: trọng số cụ thể theo phương thức ở mỗi khối, khớp attention ở dòng dư.

## Vấn đề

Cuộc tranh luận tokens hình ảnh rời rạc và liên tục đã cũ hơn LLMs. Biểu diễn liên tục (pixel thô, tiềm ẩn VAE) giữ nguyên chi tiết. tokens rời rạc (chỉ số VQ) phù hợp với từ vựng bản địa của transformer nhưng mất chi tiết ở bước quantization.

Chameleon / Emu3 rời rạc: một loss, một kiến trúc, nhưng độ trung thực của hình ảnh bị giới hạn bởi chất lượng tokenizer.

models khuếch tán diễn ra liên tục: chất lượng hình ảnh vượt trội, nhưng model riêng biệt với LLM, kỹ thuật lịch trình nhiễu phức tạp và không tích hợp rõ ràng với tạo văn bản.

Truyền máu hỏi: chúng ta có thể có cả hai không? Giữ hình ảnh liên tục, vẫn huấn luyện một model, sử dụng hai lần mất được khâu thành một bước gradient.

## Khái niệm

### Kiến trúc hai loss

Một decoder duy nhất transformer processes một chuỗi chứa:

- Văn bản tokens (rời rạc, từ từ vựng BPE).
- Các bản vá hình ảnh (các khối 16x16 pixel liên tục được chiếu vào độ mờ ẩn thông qua embedding tuyến tính - giống như đầu vào của encoder ViT).
- Thẻ `<image>` và `</image>` đánh dấu nơi tồn tại các bản vá liên tục.

Forward pass chạy một lần. loss chọn một trong hai đầu mỗi token:

- Đối với tokens văn bản: entropy chéo tiêu chuẩn trên đầu logits từ vựng.
- Đối với các bản vá hình ảnh: khuếch tán loss trên các bản vá liên tục - dự đoán nhiễu đã được thêm vào mỗi bản vá.

Sự gradient chảy qua cơ thể transformer được chia sẻ. Cả hai lần mất đều cải thiện trọng lượng được chia sẻ đồng thời.

### Mặt nạ Attention: văn bản nhân quả + hình ảnh hai chiều

Văn bản tokens phải là nhân quả - bạn không thể để một tin nhắn token tham gia vào văn bản trong tương lai hoặc giáo viên buộc phải nghỉ giải lao. Tuy nhiên, các bản vá hình ảnh đại diện cho một ảnh chụp nhanh; chúng nên tham gia vào nhau hai chiều trong cùng một khối hình ảnh.

Mặt nạ:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Được triển khai như một mặt nạ hình tam giác khối ở training và inference.

### Khuếch tán loss bên trong transformer

loss khuếch tán là tiêu chuẩn: thêm nhiễu vào bản vá hình ảnh, yêu cầu model dự đoán nhiễu (hoặc bản vá sạch, tương đương). Phiên bản của Transfusion sử dụng kết hợp dòng chảy - dự đoán trường vận tốc từ ồn ào đến sạch.

Trong thời gian training:
1. Đối với mỗi bản vá hình ảnh x0, lấy mẫu một bước thời gian ngẫu nhiên t.
2. Sample nhiễu ε, tính toán xt = (1-t) * x0 + t * ε (nội suy tuyến tính để khớp luồng).
3. transformer dự đoán v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop cùng với các tổn thất NTP văn bản từ cùng một trình tự.

Tại inference, thế hệ là:
- tokens văn bản: sampling tự hồi quy tiêu chuẩn.
- Các bản vá hình ảnh: vòng lặp sampling khuếch tán (điển hình là 10-30 bước) có điều kiện trên tokens văn bản prior.

### MMDiT: Biến thể của Stable Diffusion 3

Khuếch tán ổn định 3 (Esser và cộng sự, tháng 3 năm 2024) shipped MMDiT (Transformer khuếch tán đa phương thức) cùng thời điểm với Truyền máu. Các kiến trúc sư là anh chị em.

Sự khác biệt chính của MMDiT:

- Trọng số cụ thể theo phương thức trên mỗi khối. Mỗi khối transformer có trọng số Q, K, V và MLP riêng biệt cho các bản vá tokens văn bản so với hình ảnh. Attention là chung (phương thức chéo); mọi thứ khác đều dành riêng cho phương thức.
- Lưu lượng chỉnh lưu training. Một biến thể khớp luồng cụ thể với sampling đã biết và toán học đơn giản hơn DDPM.
- Quy mô. MMDiT là xương sống cho SD3 (các biến thể tham số 2B và 8B). Giấy truyền máu có tỷ lệ 7B.

Cả hai đều hội tụ trên cùng một ý tưởng cốt lõi: một transformer chạy NTP trên văn bản và khuếch tán trên các biểu diễn hình ảnh liên tục.

### Tại sao điều này đánh bại phong cách tắc kè hoa

Khoảng cách chất lượng giữa khuếch tán liên tục và NTP rời rạc khi tạo hình ảnh có thể đo lường được. Báo cáo giấy truyền máu:

- Với tham số 7B, đánh bại một model kiểu Chameleon cùng kích thước trên FID với 3-5 điểm.
- Không cần tokenizer training - encoder hình ảnh đơn giản hơn (Phép chiếu tuyến tính để ẩn, giống như lớp đầu vào của ViT).
- Inference có thể khử nhiễu bản vá hình ảnh song song, không giống như tokens hình ảnh tự hồi quy.

Nhược điểm: Truyền máu là một loss model kép, làm cho động lực training trở nên phức tạp hơn. Loss trọng lượng cần được điều chỉnh. Lịch trình không phù hợp giữa NTP và khuếch tán có thể khiến một đầu chiếm ưu thế.

### Những gì nằm ở hạ lưu

Janus-Pro (Bài 12.15) tinh chỉnh ý tưởng của Transfusion bằng cách tách rời encoder tầm nhìn để hiểu và tạo ra - SigLIP cho một người, VQ cho người kia - trong khi chia sẻ cơ thể transformer. Show-o (Bài 12.14) hoán đổi khuếch tán thành khuếch tán rời rạc (dự đoán mặt nạ). Gia đình thế hệ thống nhất branches nhanh chóng sau khi Truyền máu.

2026 production VLMs phát ra hình ảnh - Gemini 3 Pro, GPT-5 Claude con đường tạo hình ảnh của Opus 4.7 - gần như chắc chắn sử dụng một số hậu duệ của họ này. Thông tin chi tiết là độc quyền.

## Ứng dụng

`code/main.py` xây dựng một món đồ chơi Transfusion dựa trên một vấn đề nhỏ giống như MNIST:

- Chú thích văn bản là dãy số nguyên ngắn mô tả một chữ số (0-9).
- Hình ảnh là lưới byte 4x4.
- Một cặp phép chiếu tuyến tính có trọng lượng chia sẻ đóng vai trò là transformer thay thế; NTP loss trên văn bản, MSE loss trên các bản vá nhiễu.
- Training lặp xen kẽ hai tổn thất, attention mặt nạ rõ ràng.
- Generation tạo ra chú thích văn bản và hình ảnh 4x4 trong một forward pass.

transformer là một món đồ chơi. Hệ thống ống nước hai loss, cấu trúc mặt nạ attention và vòng lặp inference là artifacts thực sự.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-two-loss-trainer-designer.md`. Với một nhiệm vụ training đa phương thức mới (văn bản + hình ảnh, văn bản + âm thanh, văn bản + video), nó thiết kế lịch trình hai loss (trọng số loss, hình dạng mặt nạ, khối chia sẻ so với phương thức cụ thể) và gắn cờ rủi ro triển khai.

## Bài tập

1. Một model kiểu Truyền máu huấn luyện 70% tokens văn bản và 30% bản vá hình ảnh. Độ khuếch tán hình ảnh loss là ~10 lần độ lớn của NTP loss văn bản. Trọng lượng loss nào cân bằng chúng?

2. Triển khai mặt nạ hình tam giác khối cho một trình tự: `[T, T, <image>, P, P, P, P, </image>, T]`. Đánh dấu mỗi mục nhập 0 hoặc 1.

3. MMDiT có trọng số QKV dành riêng cho phương thức. Điều này thêm vào chi phí parameter bao nhiêu so với transformer được chia sẻ đầy đủ của Transfusion? Ở 7 tỷ tham số, nó có đáng không?

4. Generation: cho một prompt văn bản, model chạy NTP trong 50 tokens, sau đó đạt `<image>`, sau đó chạy khuếch tán trên 256 bản vá trên 20 bước khử nhiễu. Tổng cộng có bao nhiêu đường chuyền về phía trước?

5. Đọc giấy SD3 Phần 3. Mô tả luồng đã chỉnh lưu và lý do tại sao nó hội tụ trong ít bước inference hơn DDPM.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Hai loss training | "NTP + khuếch tán" | Một transformer duy nhất tối ưu hóa cả entropy chéo trên tokens văn bản và MSE trên các bản vá hình ảnh liên tục trong cùng một bước gradient |
| Kết hợp luồng | "Dòng chảy được chỉnh lưu" | Biến thể khuếch tán dự đoán trường vận tốc từ nhiễu đến dữ liệu sạch; toán đơn giản hơn DDPM |
| MMDiT | "DiT đa phương thức" | Kiến trúc của Stable Diffusion 3: attention chung, MLP và định mức cụ thể theo phương thức |
| Mặt nạ hình tam giác khối | "Văn bản nhân quả + hình ảnh hai chiều" | Attention mặt nạ nhân quả trên văn bản nhưng hai chiều trong vùng hình ảnh |
| Biểu diễn hình ảnh liên tục | "Không có VQ" | Các bản vá hình ảnh dưới dạng vectors có giá trị thực, không phải chỉ mục sách mã số nguyên |
| Dự đoán vận tốc | "Tham số hóa V" | Đầu ra mạng là trường vận tốc giữa nhiễu và dữ liệu, không phải bản thân nhiễu |

## Đọc thêm

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
