# MIO và Any-to-Any Streaming Models đa phương thức

> GPT-4o ships một sản phẩm mở nhất models không thể sao chép: một agent nghe giọng nói, xem video và nói lại trong thời gian thực. Câu trả lời về hệ sinh thái mở vào cuối năm 2024 là MIO (Wang và cộng sự, tháng 9 năm 2024). MIO mã hóa văn bản, hình ảnh, lời nói và âm nhạc, huấn luyện một transformer nhân quả trên các chuỗi xen kẽ và tạo ra bất kỳ phương thức nào cho bất kỳ phương thức nào. AnyGPT (Zhan và cộng sự, tháng 2 năm 2024) là bằng chứng về khái niệm; MIO là mở rộng quy mô; Unified-IO 2 (Allen AI, tháng 12 năm 2023) là người anh em họ có tầm nhìn + hành động grounding. Bài học này đọc mô hình bất kỳ - bốn tokenizers, một transformer, giải mã thân thiện với streaming.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Kiến thức tiên quyết:** Giai đoạn 12 · 11 (Tắc kè hoa), Giai đoạn 6 (Giọng nói và Âm thanh)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Thiết kế một từ vựng được chia sẻ lưu trữ văn bản, hình ảnh, lời nói và âm nhạc tokens mà không va chạm.
- So sánh SEED-Tokenizer (hình ảnh) và SpeechTokenizer residual-VQ (giọng nói) về sự đánh đổi nén + tái tạo.
- Giải thích chương trình giảng dạy bốn giai đoạn xây dựng bất kỳ thế hệ nào.
- Kể tên ba công thức nấu ăn mở bất kỳ với bất kỳ và sự đánh đổi chính của chúng: MIO, AnyGPT, Unified-IO 2.

## Vấn đề

Một model đa phương thức thống nhất rất dễ yêu cầu và khó xây dựng trên quy mô lớn. Hầu hết các hệ thống "bất kỳ" cho đến năm 2024 đều được triển khai: hình ảnh model → biểu diễn văn bản → giọng nói model → âm thanh. Mỗi bước nhảy sẽ mất thông tin, thêm độ trễ và làm phức tạp training. Video demo của GPT-4o cho thấy một giải pháp thay thế một model với phản hồi dưới giây; hệ thống mở bị tụt lại nhiều tháng.

Những thách thức về kỹ thuật:

- Tokenizers phải tồn tại cho mọi phương thức, nén không mất dữ liệu đủ để tái tạo và tạo ra tokens với tốc độ mà transformer có thể tiêu thụ.
- Một từ vựng duy nhất phải phân bổ không gian cho văn bản (32k +), hình ảnh (16k +), giọng nói (4k +), âm nhạc (8k +). Tối thiểu hơn bốn mươi nghìn mục nhập.
- Training dữ liệu phải bao gồm mọi cặp đầu vào-đầu ra (văn bản→hình ảnh, hình ảnh→giọng nói, giọng nói→hình ảnh, v.v.) hoặc model phải soạn thảo.
- Inference phải phát trực tuyến đầu ra tokens đủ nhanh để có độ trễ hội thoại (<500ms-time-to-first-audio-byte).

## Khái niệm

### Bốn tokenizers cho bốn phương thức

tokenizer stack của MIO:

- Văn bản: BPE chuẩn, từ vựng ~32000.
- Hình ảnh: SEED-Tokenizer (2023) — VAE lượng tử hóa với sách mã rời rạc, 4096 mục nhập, 32x32 tokens mỗi hình ảnh.
- Bài phát biểu: SpeechTokenizer residual-VQ (2023) — mã hóa dạng sóng 16kHz thành 8 sách mã phân cấp; Cấp độ đầu tiên là nội dung thô, cấp độ sau thêm văn xuôi và nhận dạng người nói.
- Âm nhạc: VQ dư tương tự (họ MusicGen / Encodec của Meta), 4-8 sách mã.

Mỗi phương thức tạo ra tokens số nguyên. Các tokens nhận được các phạm vi ID rời rạc trong từ vựng được chia sẻ:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Tổng cộng: ~48k từ vựng. Chiếu embedding đầu vào và đầu ra span tất cả.

### Giải mã Streaming

Tạo giọng nói sử dụng VQ dư. transformer dự đoán tokens giọng nói cơ sở (lớp 0); Một bộ định lượng dư được giải mã song song dự đoán các lớp tiếp theo. Mỗi lớp 0 token là khoảng 50ms âm thanh ở 16kHz.

Mô hình streaming:

1. Người dùng nói vào micrô; tokenizer âm thanh thời gian thực phát ra tokens giọng nói sau mỗi 50ms.
2. MIO tiêu thụ tokens khi chúng đến (prompt nạp trước + chuyển tiếp gia tăng).
3. Đầu ra tokens luồng ra như được tạo; decoder giọng nói song song chuyển đổi chúng thành các mẫu âm thanh với độ trễ ~50-150ms.
4. Thời gian đến byte âm thanh đầu tiên: ~300-500ms trong giấy MIO, gần ~250ms của GPT-4o.

Mini-Omni (arXiv: 2408.16725), GLM-4-Voice (arXiv: 2412.02612) và Moshi (arXiv: 2410.00037) là các thiết kế LLM giọng nói bổ sung streaming. Đặc biệt, Moshi đạt được 160ms khứ hồi trong một GPU.

### Chương trình giảng dạy bốn giai đoạn

Chương trình giảng dạy training của MIO:

1. Giai đoạn 1 - alignment. Kho dữ liệu cặp phương thức quy mô lớn: văn bản-hình ảnh, văn bản-lời nói, văn bản-nhạc. Mỗi cặp sử dụng phân đoạn từ vựng token của riêng mình. Huấn luyện vốn từ vựng được chia sẻ.
2. Giai đoạn 2 - xen kẽ. Các tài liệu xen kẽ đa phương thức (blog có hình ảnh + video, podcast có bảng điểm, v.v.). Huấn luyện ngữ cảnh đa phương thức.
3. Giai đoạn 3 - tăng cường giọng nói. Dữ liệu âm thanh bổ sung để nâng cao chất lượng giọng nói mà không làm mất khả năng văn bản.
4. Giai đoạn 4 - SFT. Điều chỉnh hướng dẫn trên các phương thức: VQA, phụ đề, tường thuật, đối thoại giữa giọng nói với giọng nói.

Bỏ lỡ một giai đoạn làm suy giảm các khả năng cụ thể: bỏ qua giai đoạn 2 và model mất bối cảnh đa phương thức; bỏ qua giai đoạn 3 và nói kém.

### Chuỗi suy nghĩ trực quan

MIO giới thiệu chuỗi tư duy trực quan: model phát ra tokens hình ảnh trung gian như một bước lý luận. Đối với "con mèo có đang trèo cây không?" model:

1. Phát ra `<image>` tokens hiển thị cảnh (từ hình ảnh đầu vào hoặc bản phác thảo).
2. Phát ra văn bản phân tích bản phác thảo.
3. Phát ra câu trả lời cuối cùng.

Hình ảnh trung gian được hiển thị đóng vai trò như một bảng cào. Benchmarks cải thiện các nhiệm vụ suy luận không gian. Ý tưởng phản ánh chain-of-thought cho lý luận văn bản.

### Đối thủ cạnh tranh trong any-to-any

- AnyGPT (arXiv:2402.12226): 4 phương thức (văn bản, hình ảnh, lời nói, âm nhạc), thiết kế tương tự.
- Unified-IO 2 (arXiv: 2312.17172): thêm đầu ra hành động tầm nhìn, độ sâu, pháp tuyến. Đa dạng nhiệm vụ hơn, quy mô nhỏ hơn.
- NExT-GPT (arXiv: 2309.05519): decoders khuếch tán LLM + phương thức cụ thể. Không phải là cách tiếp cận một model.
- CoDi (arXiv:2305.11846): khuếch tán có thể kết hợp; bất kỳ với bất kỳ thông qua tiềm ẩn được chia sẻ.

MIO gần nhất với thuần túy token bất kỳ. AnyGPT là tổ tiên khái niệm của nó.

### Ngân sách độ trễ

Đối với một sản phẩm đàm thoại, độ trễ của mọi thành phần đều quan trọng:

- Mic sang tokens âm thanh: ~50ms.
- Điền trước (tokens âm thanh + lịch sử): ~100ms trên model 8B.
- token đầu ra đầu tiên: ~50ms.
- VQ dư song song + decoder giọng nói: ~100-150ms.

Tổng thời gian đến byte âm thanh đầu tiên: tối thiểu ~300ms. GPT-4o tuyên bố ~250ms. Moshi tuyên bố 160ms. MIO/AnyGPT nằm trong phạm vi 400-600ms cho mỗi benchmarks công cộng.

### Tại sao any-to-any vẫn khó khăn

Ngay cả vào năm 2026, mở bất kỳ models đường mòn nào cũng đóng trên hai trục:

- Chất lượng giọng nói. tokenizer VQ dư bị tổn thất; giọng nói đàm thoại nghe có vẻ rô bốt so với giọng nói class của ElevenLabs.
- Lý luận đa phương thức. Yêu cầu model "hát về những gì bạn thấy" vẫn thất bại thường xuyên hơn so với các nhiệm vụ thuần túy.

Đây là những vấn đề nghiên cứu mở. Qwen3-Omni (Bài 12.20) là nỗ lực mở tiên tiến nhất vào năm 2025.

## Ứng dụng

`code/main.py`:

- Xác định phân bổ từ vựng bốn phương thức và in nó.
- Định tuyến danh sách các đầu vào đa phương thức (văn bản, hình ảnh, clip âm thanh, nhạc) thông qua bộ định tuyến tokenizer.
- Mô phỏng streaming giải mã cho phản hồi chuyển văn bản thành giọng nói với tính năng đếm độ trễ.
- Tính toán thời gian dự kiến đến byte âm thanh đầu tiên với độ trễ encoder, nạp trước và decoder.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-any-to-any-pipeline-auditor.md`. Với thông số kỹ thuật sản phẩm đàm thoại (phương thức vào, phương thức ra, mục tiêu độ trễ), nó kiểm tra các lựa chọn thiết kế dòng MIO và tính toán ngân sách độ trễ.

## Bài tập

1. Sản phẩm của bạn chấp nhận đầu vào giọng nói và trả về đầu ra giọng nói. Mục tiêu ngân sách độ trễ từ đầu đến cuối là bao nhiêu? Liệt kê các thành phần dành thời gian.

2. SpeechTokenizer residual-VQ sử dụng 8 codebook. Đề xuất lý do tại sao giải mã song song các mức dư là cần thiết (so với tuần tự) và tiết kiệm độ trễ mà nó mang lại.

3. Từ vựng của bạn có 32k văn bản + 4k hình ảnh + 4k giọng nói. Thêm nhạc 8k và ~10 dấu phân cách. Chi phí parameter ma trận embedding tại hidden dim 4096 là bao nhiêu?

4. Chuỗi tư tưởng thị giác phát ra một hình ảnh trung gian. Những loại câu hỏi nào có lợi? Những loại nào bị tổn thương bởi tokens thừa?

5. Đọc Moshi (arXiv: 2410.00037). Mô tả kỹ thuật "độc thoại bên trong" của nó và so sánh với chuỗi suy nghĩ trực quan của MIO.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Bất kỳ đối với bất kỳ | "in/out đa phương thức" | Một model duy nhất chấp nhận và phát ra văn bản, hình ảnh, lời nói và âm nhạc theo bất kỳ hướng nào |
| VQ dư | "Bài phát biểu tokenizer stack" | tokenization nhiều codebook nơi mỗi lớp thêm thông tin; lớp cơ sở là nội dung, các lớp sau là prosody |
| Tokenizer hạt giống | "Mã hình ảnh" | tokenizer hình ảnh rời rạc với sổ mã 4096 mục được MIO sử dụng |
| Chuỗi suy nghĩ trực quan | "Bàn cào trực quan" | model tạo ra một hình ảnh trung gian như một bước suy luận trước câu trả lời cuối cùng của nó |
| Thời gian đến byte âm thanh đầu tiên | "TTFAB" | Độ trễ từ giọng nói của người dùng đến đầu ra âm thanh đầu tiên; <500ms cho cảm giác đàm thoại |
| Chương trình giảng dạy bốn giai đoạn | "Công thức Training" | Alignment -> xen kẽ -> tăng cường giọng nói -> SFT, theo thứ tự đó |

## Đọc thêm

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
