# Models ngôn ngữ âm thanh: The Whisper to Audio Flamingo 3 Arc

> Whisper (Radford và cộng sự, tháng 12 năm 2022) đã giải quyết nhận dạng giọng nói - 680 nghìn giờ nói đa ngôn ngữ được giám sát yếu, một decoder transformer encoder đơn giản, một benchmark khiến mọi bản phát hành ASR tiếp theo đều trích dẫn nó. Nhưng sự công nhận không phải là lý luận. Hỏi "nhạc cụ nào trong bản ghi âm này" hoặc "người nói đang thể hiện cảm xúc gì" hoặc "điều gì đã xảy ra ở phút thứ 3" đòi hỏi sự hiểu biết về âm thanh, không phải phiên âm. Qwen-Audio, SALMONN, LTU và Audio Flamingo 3 của NVIDIA (AF3, tháng 7 năm 2025) dần dần xây dựng stack đó: giữ Whisper-class encoders, bắt vít trên Q-formers, huấn luyện dữ liệu hướng dẫn văn bản âm thanh, thêm lý luận chain-of-thought. Bài học này đi theo vòng cung.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Kiến thức tiên quyết:** Giai đoạn 6 (Giọng nói và Âm thanh), Giai đoạn 12 · 03 (Q-Cũ)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Tính toán quang phổ log-Mel từ dạng sóng: windowing, FFT, ngân hàng bộ lọc, chuyển đổi log.
- So sánh các tùy chọn encoder: Whisper encoder, BEATs, AF-Whisper hybrid. Khi mỗi người thắng.
- Xây dựng một Q âm thanh trước đây: N truy vấn có thể học được tham gia chéo các bản vá quang phổ.
- Giải thích LLM training âm thanh theo tầng (Thì thầm rồi LLM) so với đầu cuối: tại sao đầu cuối lại mở rộng quy mô tốt hơn cho lý luận.

## Vấn đề

Nhận dạng giọng nói đã được giải quyết bởi Whisper. OCR-of-audio là một mặt hàng. Nhưng "hàng hóa" dừng lại ở phiên âm. Nếu model không thể suy luận về những gì nó nghe được - thời gian, loa, cảm xúc, cấu trúc âm nhạc, âm thanh môi trường - chỉ phiên âm không thể thúc đẩy features sản phẩm.

Ba tuyến đường rõ ràng:

1. Cascade: Whisper phiên âm, LLM lý do trên bản ghi. Hoạt động cho các tình huống thuần túy. Không thành công đối với âm nhạc, âm thanh môi trường, chồng chéo nhiều loa, cảm xúc.

2. LLM âm thanh đầu cuối: encoder âm thanh cung cấp tokens âm thanh trực tiếp vào LLM, bỏ qua phiên âm. Bảo toàn thông tin âm thanh (cảm xúc, loa, môi trường). Cần dữ liệu training mới.

3. Kết hợp: encoder âm thanh + decoder văn bản có thể vừa phiên âm vừa có thể suy luận. Qwen-Audio và Audio Flamingo chọn tuyến đường này.

## Khái niệm

### Quang phổ Log-Mel: feature đầu vào

Mọi encoder âm thanh đều bắt đầu với cùng một feature: quang phổ log-Mel.

1. Lấy mẫu lại đến 16 kHz.
2. Biến đổi Fourier trong thời gian ngắn với windows 25ms, nhảy 10ms.
3. Lấy độ lớn của kết quả FFT.
4. Áp dụng các ngân hàng bộ lọc Mel (thường là 80 bộ lọc cách nhau 0-8000 Hz) để cong theo tần số nhận thức.
5. Nén nhật ký (log(1 + x)) cho dải động.

Kết quả: một mảng hình dạng 2D (T, 80) trong đó T là số khung thời gian. Đối với clip dài 30 giây ở tốc độ khung hình 100 Hz: (3000, 80).

### Whisper's encoder

Whisper's encoder là một transformer kiểu ViT 12 lớp xử lý quang phổ log-Mel dưới dạng một chuỗi các khung thời gian. Đầu ra: một vector trạng thái ẩn cho mỗi khung thời gian.

Đối với ASR, decoder của Whisper là một cross-attention transformer tạo văn bản tokens điều kiện trên đầu ra encoder. Tiêu chuẩn encoder-decoder.

Đối với ALM (âm thanh-LLMs), bạn muốn đầu ra encoder làm đầu vào cho một LLM khác. Mẫu: Whisper encoder đóng băng, Q-former có thể huấn luyện, LLM đóng băng hoặc điều chỉnh.

### BEAT và encoders dành riêng cho âm thanh

Whisper được huấn luyện về dữ liệu chủ yếu là giọng nói. Nó yếu hơn đối với âm nhạc và âm thanh môi trường.

BEATs (Chen và cộng sự, 2022) là một transformer tự giám sát được huấn luyện trên AudioSet. Ghi lại âm nhạc và âm thanh môi trường tốt hơn Whisper ở cùng một parameter số.

AF-Whisper (Audio Flamingo 3's hybrid): concat Whisper + BEAT features làm đầu vào âm thanh. Whisper mang tín hiệu ngôn ngữ, BEAT mang tín hiệu âm thanh.

### Âm thanh Q-former

Mẫu tương tự như Q-former trực quan của BLIP-2. Một số truy vấn cố định có thể học được (thường là 32 hoặc 64) tham dự chéo trên các khung đầu ra của encoder âm thanh. Các truy vấn trở thành âm thanh tokens LLM tiêu thụ.

Training alignment giai đoạn: Q-former một mình, mất tương phản + phụ đề trên các cặp âm thanh-văn bản (AudioCaps, Clotho). Giai đoạn hướng dẫn: đầu cuối, giải phóng LLM, huấn luyện về dữ liệu lệnh.

### Vòng cung - SALMONN, Qwen-Audio, AF3

SALMONN (Tang và cộng sự, 2023): Thì thầm + BEATs + Q-former + LLaMA. LLM âm thanh mở đầu tiên với khả năng suy luận nghiêm túc. Benchmarks trên chương trình MMAU ~0,55 tổng hợp.

Qwen-Audio (Chu và cộng sự, 2023): kiến trúc tương tự, được huấn luyện trên dataset phong phú hơn, được điều chỉnh cho cuộc đối thoại nhiều lượt. MMAU ~0.60.

LTU — Nghe, Suy nghĩ, Hiểu (Gong và cộng sự, 2023): dữ liệu suy luận rõ ràng, tập trung vào chain-of-thought hơn các clip âm thanh. Nhỏ hơn nhưng tập trung hơn.

Audio Flamingo 3 (Goel et al., tháng 7 năm 2025): SOTA mở hiện tại. 8B LLM đường trục (Qwen2 7B), Whisper-large encoder concat BEATs, 64-query Q-former training trên 1M + cặp lệnh âm thanh-văn bản. MMAU 0.72, phù hợp với biên giới độc quyền trên một số nhiệm vụ phụ.

AF3 cũng giới thiệu chain-of-thought theo yêu cầu cho âm thanh: model có thể tùy chọn phát ra tokens suy nghĩ ("hãy để tôi xác định các nhạc cụ trước: ...") trước câu trả lời cuối cùng. Accuracy với các nhiệm vụ lý luận phức tạp nâng 3-5 điểm khi tư duy được kích hoạt.

### Xếp tầng so với đầu đến cuối

pipeline xếp tầng:

1. Whisper phiên âm âm thanh → văn bản.
2. LLM lý do hơn là văn bản.

Hoạt động hoàn hảo cho "tóm tắt podcast này". Không thành công vì:
- "Tâm trạng của bài hát này là gì?" - tâm trạng nằm ở âm thanh, không phải lời nói.
- "Ai đang nói, Alice hay Bob?" - yêu cầu nhận dạng người nói.
- "Vụ nổ xảy ra vào giây nào?" - tạm thời grounding bị mất trong văn bản.
- "Đây là âm thanh thật hay được tạo ra?" - phát hiện deepfake cần features âm thanh.

Đầu cuối bảo toàn tín hiệu âm thanh. Qwen-Audio và AF3 xử lý âm nhạc, môi trường và cảm xúc nguyên bản.

### Công thức production 2026

Đối với một sản phẩm hiểu âm thanh mới:

- Xếp tầng nếu: phiên âm là mục tiêu, không có âm nhạc, không có cảm xúc inference.
- AF3 / Qwen-Audio-family nếu: âm nhạc, cảm xúc, nhiều loa hoặc suy luận âm thanh phức tạp.

Xếp tầng rẻ hơn và đơn giản hơn. End-to-end có nhiều khả năng hơn.

### MMAU — lý luận âm thanh benchmark

MMAU (Hiểu âm thanh đa phương thức lớn) là benchmark lý luận âm thanh 2024-2025:

- 10.000 cặp QA âm thanh-văn bản trên giọng nói, âm nhạc, âm thanh môi trường.
- Bao gồm phân loại, suy luận thời gian, lý luận nhân quả, QA kết thúc mở.
- Kiểm tra những gì xếp tầng pipelines bỏ lỡ một cách có hệ thống.

Mở SOTA (AF3) ở mức 0,72; biên giới độc quyền ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Khoảng cách nhỏ hơn so với delta mở và đóng của VideoMME, cho thấy LLMs âm thanh đang trưởng thành.

## Ứng dụng

`code/main.py`:

- Thực hiện tính toán quang phổ log-Mel trong stdlib: windowing, DFT ngây thơ, ngân hàng bộ lọc Mel.
- Bộ xương âm thanh Q-cũ: cho encoder khung đầu ra, tính toán Q, K, V, attention và phát ra N tokens.
- So sánh xếp tầng và đầu cuối trên một nhiệm vụ đồ chơi.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-audio-llm-pipeline-picker.md`. Được giao một tác vụ âm thanh (phiên âm, gắn thẻ nhạc, inference cảm xúc, phân loại nhiều loa, phân loại môi trường), nó chọn AF3 xếp tầng, đầu cuối hoặc kết hợp.

## Bài tập

1. Tính toán kích thước quang phổ log-Mel cho clip 30 giây ở 16kHz, cửa sổ 25ms, bước nhảy 10ms, 80 thùng Mel. Điều này thay đổi như thế nào ở 48kHz?

2. Tại sao Whisper hoạt động kém hiệu quả về âm nhạc? BEAT ghi lại features âm thanh nào mà Whisper không ghi lại?

3. Audio Q-former với 64 truy vấn so với 32: 64 độ phức tạp của nhiệm vụ nào được đền đáp? 32 Tiết kiệm điện toán để làm gì?

4. Đọc AF3 Phần 4 về tư duy theo yêu cầu. Đề xuất ba nhiệm vụ âm thanh mà chain-of-thought giúp ích nhiều nhất.

5. Thực hiện pipeline diarization tối thiểu bằng cách sử dụng đầu ra của AF3. Làm thế nào để bạn báo hiệu thay đổi loa?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Quang phổ Log-Mel | "Mel features" | Mảng 2D (thời gian, tần số) của các giá trị log-magnitude sau ngân hàng bộ lọc Mel |
| Âm thanh Q-former | "Bộ nhận biết âm thanh" | Cross-attention tắc nghẽn từ đầu ra encoder âm thanh đến các truy vấn có độ dài cố định cung cấp cho LLM |
| Xếp tầng | "ASR rồi LLM" | Pipeline nơi Whisper phiên âm và một văn bản LLM lý do; mất thông tin âm thanh |
| Đầu cuối | "Âm thanh-LLM" | Âm thanh features vào LLM trực tiếp thông qua Q-former; bảo toàn tín hiệu âm thanh |
| NHỊP ĐẬP | "Bộ âm thanh encoder" | SSL transformer huấn luyện trên AudioSet; Âm nhạc mạnh mẽ + âm thanh môi trường |
| MMAU | "Băng ghế lý luận âm thanh" | 10k cặp QA trên lời nói, âm nhạc, môi trường; Tiêu chuẩn EVAL 2024 |
| Tư duy theo yêu cầu | "Âm thanh CoT" | Model có thể tùy ý phát ra tokens lý luận trước khi trả lời cuối cùng, nâng accuracy 3-5 điểm |

## Đọc thêm

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
