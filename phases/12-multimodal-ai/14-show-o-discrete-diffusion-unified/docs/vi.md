# Show-o và Discrete-Diffusion Unified Models

> Truyền máu kết hợp các đại diện liên tục và rời rạc. Show-o (Xie và cộng sự, tháng 8 năm 2024) đi theo hướng khác: văn bản tokens sử dụng dự đoán nhân quả token tiếp theo, hình ảnh tokens sử dụng khuếch tán rời rạc được che giấu theo tinh thần của MaskGIT. Cả hai đều ngồi bên trong một transformer với mặt nạ attention lai. Kết quả hợp nhất VQA, chuyển văn bản thành hình ảnh, vẽ và tạo phương thức hỗn hợp trên một xương sống, một tokenizer cho mỗi phương thức, một công thức loss (token tiếp theo mở rộng sang dự đoán mặt nạ). Bài học này đi theo thiết kế Show-o - tại sao khuếch tán rời rạc mặt nạ là một trình tạo hình ảnh song song, vài bước - và tương phản với Transfusion và Emu3.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, masked-discrete-diffusion sampler)
**Kiến thức tiên quyết:** Giai đoạn 12 · 13 (Truyền máu)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích sự khuếch tán rời rạc được che nạp: lịch trình mặt nạ tokens đồng nhất sau đó yêu cầu transformer khôi phục chúng.
- So sánh giải mã hình ảnh song song (Show-o, MaskGIT) với giải mã hình ảnh tự hồi quy (Chameleon, Emu3) về tốc độ và chất lượng.
- Đặt tên cho ba tác vụ mà Show-o xử lý trong một checkpoint: T2I, VQA, vẽ hình ảnh.
- Chọn lịch trình che (cosin, tuyến tính, cắt bớt) và lý do về ảnh hưởng của nó đối với chất lượng mẫu.

## Vấn đề

Hai loss training của Transfusion hoạt động nhưng có động lực học phức tạp hơn - sự khuếch tán liên tục loss sống trên một thang số khác với NTP loss rời rạc. Cân bằng trọng lượng loss là một tìm kiếm hyperparameter. Kiến trúc hiệu quả nhưng phức tạp.

Câu trả lời của Show-o: giữ cả hai phương thức rời rạc (như tắc kè hoa), nhưng tạo hình ảnh song song thông qua khuếch tán rời rạc được che giấu thay vì tuần tự. Mục tiêu training trở thành một dự đoán token được che giấu duy nhất khái quát hóa dự đoán token tiếp theo một cách tự nhiên.

## Khái niệm

### Khuếch tán rời rạc mặt nạ (MaskGIT)

Thủ thuật MaskGIT ban đầu của Chang et al. (2022) rất thanh lịch. Bắt đầu từ một hình ảnh được che hoàn toàn (mỗi token là id `<MASK>` đặc biệt). Ở mỗi bước, dự đoán song song tất cả các tokens được che giấu, sau đó giữ top-K dự đoán tự tin nhất và che lại rest. Sau ~8-16 lần lặp, tất cả các tokens đều được điền. Lịch trình về số lượng tokens để vạch mặt mỗi bước được điều chỉnh - lịch trình cosine hoạt động tốt.

Training rất đơn giản: lấy mẫu tỷ lệ mặt nạ đồng nhất từ [0, 1], áp dụng nó cho tokens VQ của hình ảnh, huấn luyện các transformer để khôi phục các mặt nạ bị che nắng. Chính xác những gì BERT đã làm cho văn bản, được chia tỷ lệ để tạo hình ảnh.

### Show-o: một transformer, mặt nạ lai

Show-o đặt MaskGIT bên trong một model transformer ngôn ngữ nhân quả. Mặt nạ attention là:

- Văn bản tokens: nhân quả (LLM tiêu chuẩn).
- Hình ảnh tokens: hoàn toàn hai chiều trong khối hình ảnh (để tokens được che có thể nhìn thấy mọi hình ảnh khác token trong quá trình dự đoán).
- Chuyển văn bản thành hình ảnh: văn bản tham gia vào hình ảnh prior, hình ảnh tham gia vào văn bản prior.

Training xen kẽ giữa:
1. NTP tiêu chuẩn trên chuỗi văn bản.
2. Mẫu T2I: văn bản → hình ảnh với tokens hình ảnh được che giấu, loss dự đoán token mặt nạ.
3. Mẫu VQA: hình ảnh → văn bản với văn bản được che tokens (thực sự chỉ là NTP).

loss thống nhất là entropy chéo trên `<MASK>` tokens, bao gồm cả NTP văn bản (chỉ token cuối cùng là "mặt nạ") và khuếch tán hình ảnh (tập con ngẫu nhiên được che giấu).

### sampling song song

Show-o tạo ra một hình ảnh trong ~16 bước thay vì ~1000 (tự hồi quy mỗi token) hoặc ~20 (khuếch tán). Ở mỗi bước, dự đoán song song tất cả các tokens mặt nạ; commit top-K tự tin; lặp lại.

So sánh:
- Tắc kè hoa / Emu3 (tự hồi quy trên tokens): N_tokens đường chuyền về phía trước, thường là 1024-4096 cho mỗi hình ảnh.
- Truyền máu (khuếch tán liên tục): ~20 bước, mỗi bước transformer trọn vẹn.
- Show-o (khuếch tán rời rạc mặt nạ): ~16 bước, mỗi bước transformer đầy đủ.

Show-o nhanh hơn Tắc kè hoa ở tỷ lệ tương tự models, gần bằng với số bước Truyền máu với chi phí mỗi bước thấp hơn (logits từ vựng rời rạc so với loss MSE liên tục).

### Nhiệm vụ trong một checkpoint

Show-o hỗ trợ bốn tác vụ ở inference, được chọn theo định dạng prompt:

- Tạo văn bản: đầu ra văn bản tự hồi quy tiêu chuẩn.
- VQA: hình ảnh vào, văn bản ra.
- T2I: văn bản vào, hình ảnh ra thông qua khuếch tán rời rạc được che giấu.
- Inpainting: hình ảnh có một số tokens được che đậy, điền vào.

Khả năng inpainting đến miễn phí từ training dự đoán mặt nạ. Che một vùng của lưới VQ-token, cung cấp rest cùng với prompt văn bản, dự đoán tokens được che nắng.

### Lịch đeo khẩu trang

Lịch trình của bao nhiêu tokens để vạch mặt mỗi bước định hình chất lượng. Show-o khuyến nghị cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Ở bước 0, tất cả tokens được che (tỷ lệ 1.0). Ở bước T, không ai đeo mặt nạ. Cosine tập trung khối lượng vào các tỷ lệ tầm trung nơi dự đoán là nhiều thông tin nhất. Lịch trình tuyến tính cũng hoạt động nhưng ổn định nhanh hơn.

### Hiển thị-o2

Tỷ lệ Show-o2 (theo dõi năm 2025, arXiv 2506.15564) Show-o: cơ sở LLM lớn hơn, tokenizer tốt hơn, lịch trình mặt nạ được cải thiện. Cùng một mô hình kiến trúc.

### Vị trí của Show-o

Trong phân loại năm 2026:

- tokens rời rạc + NTP: Tắc kè hoa, Emu3. Đơn giản nhưng chậm inference.
- Khuếch tán tokens rời rạc + mặt nạ: Show-o, MaskGIT, LlamaGen, Muse. Song song sampling, vẫn bị mất tokenizer.
- Liên tục + khuếch tán: Truyền máu, MMDiT, DiT. Chất lượng cao nhất, training phức tạp hơn.
- Kết hợp liên tục + luồng trong một VLM: JanusFlow, InternVL-U. Mới nhất.

Chọn theo nhiệm vụ: Hiển thị khi bạn muốn T2I + inpainting + VQA trong một model mở với tốc độ hợp lý; Truyền máu khi chất lượng là tối quan trọng và bạn có thể mua được hệ thống ống nước hai loss.

## Ứng dụng

`code/main.py` mô phỏng Show-o sampling:

- Một lưới đồ chơi gồm 16 VQ tokens.
- Một "transformer" giả dự đoán logits dựa trên một prompt và tokens hiện đang được vạch mặt.
- Mặt nạ song song sampling hơn 8 bước với lịch trình cosin.
- In các trạng thái trung gian (tiến hóa mẫu mặt nạ) và tokens cuối cùng.

Chạy nó, xem mặt nạ hòa tan từng bước.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-unified-gen-model-picker.md`. Cho một sản phẩm cần cả sự hiểu biết (VQA, chú thích) và thế hệ (T2I, inpainting) với hạn chế trọng lượng mở, lựa chọn giữa gia đình Show-o, gia đình Transfusion/MMDiT và gia đình Emu3 / Chameleon với sự đánh đổi cụ thể.

## Bài tập

1. Các mẫu khuếch tán rời rạc được che mặt trong ~16 bước. Tại sao không phải là 1? Điều gì sẽ xảy ra nếu bạn vạch mặt mọi thứ ở bước 0?

2. Inpainting là miễn phí với sự khuếch tán mặt nạ. Đề xuất một trường hợp sử dụng sản phẩm (thực tế hoặc giả định) trong đó bản vẽ của Show-o đánh bại một model chuyên gia.

3. Lịch trình cosin so với lịch trình tuyến tính: trace số lượng tokens không che mặt nạ trên mỗi bước cho T = 8. Cái nào cân bằng hơn?

4. Hình ảnh Show-o 512x512 là 1024 tokens. Tại từ vựng K=16384, model phát ra 1024 * log2(16384) = 14.336 bit (~1,75 KiB) dữ liệu. Đầu ra khuếch tán ổn định 512 * 512 * 24 bit = 6,291,456 bit (~768 KiB) pixel thô. Tỷ số nén là gì và nó mua chất lượng như thế nào?

5. Đọc LlamaGen (arXiv: 2406.06525). Hình ảnh tự hồi quy class điều kiện của LlamaGen model khác với cách tiếp cận mặt nạ của Show-o như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Khuếch tán rời rạc mặt nạ | "Phong cách MaskGIT" | Training để dự đoán tokens đeo mặt nạ; Tại inference, lặp đi lặp lại những dự đoán đáng tin cậy nhất |
| Lịch trình cosin | "Lịch trình gỡ mặt nạ" | Tỷ lệ mặt nạ giảm qua inference bước; tập trung tăng trưởng tự tin ở tầm trung |
| Giải mã song song | "Tất cả tokens cùng một lúc" | Mỗi bước dự đoán toàn bộ chuỗi tokens được che giấu trong một forward pass, sau đó commits top-K |
| attention lai | "Nhân quả + hai chiều" | Mặt nạ nhân quả trên tokens văn bản và hai chiều trong khối hình ảnh |
| Sơn | "Tạo điền" | Điều kiện trên một hình ảnh có một số tokens được che mặt, dự đoán những hình ảnh bị thiếu; Không có mục tiêu training |
| Tỷ lệ cam kết | "Top-K mỗi bước" | Có bao nhiêu tokens được tuyên bố là "hoàn thành" cho mỗi lần lặp; Kiểm soát inference và đánh đổi chất lượng |

## Đọc thêm

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
