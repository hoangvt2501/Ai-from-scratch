# Chống giả mạo giọng nói và tạo hình mờ âm thanh - ASVspoof 5, AudioSeal, WaveVerify

> Nhân bản giọng nói shipped nhanh hơn phòng thủ. Hệ thống giọng nói production 2026 cần hai thứ: máy dò (AASIST, RawNet2) phân loại giọng nói thật và giả và hình mờ (AudioSeal) tồn tại sau khi nén và chỉnh sửa. Ship cả hai hoặc không ship nhân bản giọng nói.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 06 (Nhận dạng loa), Giai đoạn 6 · 08 (Nhân bản giọng nói)
**Thời lượng:** ~75 phút

## Vấn đề

Ba biện pháp phòng thủ liên quan:

1. **Chống giả mạo / phát hiện deepfake.** Cho một đoạn âm thanh, nó là tổng hợp hay thật? ASVspoof benchmarks (ASVspoof 2019 → 2021 → 5) là tiêu chuẩn vàng.
2. **Hình mờ âm thanh.** Nhúng tín hiệu không thể nhận thấy vào âm thanh được tạo ra mà máy dò có thể trích xuất sau này. AudioSeal (Meta) và WavMark là các tùy chọn mở.
3. **Nguồn gốc được xác thực.** Ký mật mã các tệp âm thanh + siêu dữ liệu. C2PA / Sáng kiến xác thực nội dung.

Phát hiện xử lý những kẻ thù không hợp tác. Watermarking xử lý việc tuân thủ - âm thanh do AI tạo phải có thể được xác định như vậy. Cả hai đều được yêu cầu vào năm 2026.

## Khái niệm

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — the 2024-2025 benchmark

Thay đổi lớn nhất so với prior phiên bản:

- **Dữ liệu cộng đồng** (không sạch studio) — điều kiện thực tế.
- **~2000 loa **(so với ~100 trước đây).
- **32 thuật toán tấn công.** TTS + chuyển đổi giọng nói + nhiễu loạn đối nghịch.
- **Hai rãnh.** Phát hiện độc lập biện pháp đối phó (CM); ASV mạnh mẽ giả mạo (SASV) cho các hệ thống sinh trắc học.

Hiện đại trên ASVspoof 5: ~7.23% EER. Trên ASVspoof 2019 LA cũ hơn: 0.42% EER. Triển khai trong thế giới thực: mong đợi 5-10% EER trên các clip trong tự nhiên.

### AASIST và RawNet2 — phát hiện model gia đình

**AASIST** (2021, cập nhật đến năm 2026). Đồ thị attention trên features quang phổ. Nhiệm vụ đối phó SOTA hiện tại trên ASVspoof 5.

**RawNet2.** Giao diện người dùng tích chập trên dạng sóng thô + đường trục TDNN. Đường cơ sở đơn giản hơn; vẫn cạnh tranh với fine-tuning.

**NeXt-TDNN + SSL features.** Biến thể 2025: kiểu ECAPA + WavLM features + loss tiêu cự. Đạt được EER 0.42% trên ASVspoof 2019 LA.

### AudioSeal — mặc định hình mờ năm 2024

**AudioSeal** của Meta (Jan 2024, v0.2 Dec 2024). Thiết kế chính:

- **Bản địa hóa.** Phát hiện hình mờ trên mỗi khung hình ở độ phân giải mẫu 16 kHz (1/16000 giây).
- **Máy phát điện + máy dò được huấn luyện chung.** Máy phát điện học cách nhúng tín hiệu không nghe được; Detector học cách tìm nó thông qua tăng cường.
- **Mạnh mẽ.** Tồn tại với nén MP3 / AAC, EQ, chuyển tốc độ ±10%, hỗn hợp nhiễu +10 dB SNR.
- **Nhanh.** Máy dò chạy ở tốc độ 485× thời gian thực; Nhanh hơn 1000× so với WavMark.
- **Dung lượng.** payload 16-bit (có thể mã hóa ID model, dấu thời gian thế hệ, ID người dùng) có thể nhúng trong mỗi lời nói.

### Đánh dấu Wav

Đường cơ sở mở trước AudioSeal. Mạng nơ-ron đảo ngược, 32 bits/sec. Vấn đề:

- Đồng bộ hóa brute-force chậm.
- Có thể được loại bỏ bằng nhiễu Gaussian hoặc nén MP3.
- Không thân thiện với thời gian thực.

### WaveVerify (Tháng Bảy 2025)

Giải quyết các điểm yếu của AudioSeal - cụ thể là các thao tác thời gian (đảo ngược, tốc độ). Sử dụng máy phát điện dựa trên FiLM + Máy dò hỗn hợp chuyên gia. Cạnh tranh với AudioSeal trên các cuộc tấn công tiêu chuẩn; xử lý các chỉnh sửa thời gian.

### Lỗ hổng mà đối thủ khai thác

Từ AudioMarkBench: "khi thay đổi cao độ, tất cả các hình mờ hiển thị Bit Recovery Accuracy dưới 0,6, cho thấy việc xóa gần như hoàn toàn." **Pitch-shift là cuộc tấn công phổ biến.** Không có hình mờ 2026 nào hoàn toàn mạnh mẽ để sửa đổi cao độ mạnh mẽ. Đây là lý do tại sao bạn cần phát hiện (AASIST) cùng với hình mờ.

### C2PA / Sáng kiến xác thực nội dung

Không phải là một kỹ thuật ML - một định dạng hiển nhiên. Các tệp âm thanh mang siêu dữ liệu được ký bằng mật mã về công cụ tạo, tác giả, ngày tháng. Audobox / Sử dụng liền mạch nó. Tốt cho nguồn gốc; không làm gì nếu kẻ xấu mã hóa lại và loại bỏ siêu dữ liệu.

## Tự xây dựng

### Bước 1: một máy dò feature quang phổ đơn giản (đồ chơi)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Lời nói tổng hợp thường có năng lượng tần số cao phẳng bất thường. Production máy dò sử dụng AASIST, không phải cái này. Nhưng trực giác vẫn còn.

### Bước 2: Nhúng + phát hiện AudioSeal

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Bước 3: đánh giá - EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Bước 4: tích hợp production

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Mỗi thế hệ ships: (1) hình mờ, (2) tệp kê khai có chữ ký, (3) nhật ký kiểm tra tuân thủ policy lưu giữ.

## Ứng dụng

| Trường hợp sử dụng | Phòng thủ |
|----------|---------|
| Shipping TTS / nhân bản giọng nói | AudioSeal được nhúng trên mọi đầu ra (không thể thương lượng) |
| Mở khóa bằng giọng nói sinh trắc học | Hòa tấu AASIST + ECAPA; Thử thách Liveness |
| Phát hiện gian lận tổng đài cuộc gọi | AASIST trên 20% mẫu cuộc gọi đến |
| Tính xác thực của podcast | Ký C2PA khi tải lên, AudioSeal nếu AI tạo |
| Máy dò nghiên cứu / training | ASVspoof 5 bộ train/dev/eval |

## Cạm bẫy

- **Hình mờ mà không có máy dò nào chạy.** Vô nghĩa. Ship máy dò trong CI của bạn.
- **Phát hiện mà không cần hiệu chuẩn.** AASIST được huấn luyện về ASVspoof LA overfits; accuracy trong thế giới thực giảm. Hiệu chỉnh trên miền của bạn.
- **Khoảng cách thay đổi cao độ.** Thay đổi cao độ mạnh mẽ loại bỏ hầu hết các hình mờ. Có dự phòng phát hiện.
- **Siêu dữ liệu dải và lưu trữ lại.** C2PA có thể bỏ qua một cách tầm thường bằng cách mã hóa lại. Luôn thêm mật mã + phòng thủ nhận thức (hình mờ) với nhau.
- **Sự sống động như phát hiện.** Yêu cầu người dùng nói một cụm từ ngẫu nhiên. Ngăn chặn các cuộc tấn công phát lại nhưng không nhân bản thời gian thực.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-spoof-defender.md`. Chọn model phát hiện, hình mờ, bản kê khai xuất xứ và cẩm nang hoạt động để triển khai tạo giọng nói.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Máy dò đồ chơi + hình mờ đồ chơi embed/detect trên âm thanh tổng hợp.
2. **Trung bình.** Cài đặt `audioseal`, nhúng payload 16 bit vào đầu ra TTS, giải mã lại. Làm hỏng âm thanh bằng nhiễu và đo lường Accuracy khôi phục bit.
3. **Khó.** Fine-tune RawNet2 hoặc AASIST trên ASVspoof 2019 LA. Đo EER. Kiểm tra trên một bộ clip được tạo bởi F5-TTS - xem khả năng phát hiện OOD giảm như thế nào.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Giả mạo ASV | Các benchmark | Thử thách hai năm một lần; 2024 = ASVspoof 5. |
| CM (biện pháp đối phó) | Máy dò | Bộ phân loại: giọng nói thực so với tổng hợp / chuyển đổi. |
| SASV | Phiên bản loa + CM | Tích hợp sinh trắc học + phát hiện giả mạo. |
| Con dấu âm thanh | Hình mờ meta | Bản địa hóa, payload 16 bit, nhanh hơn 485× so với WavMark. |
| Phục hồi bit Accuracy | Sống sót hình mờ | Phần nhỏ của payload bit được phục hồi sau khi tấn công. |
| C2PA | Bản kê khai xuất xứ | Siêu dữ liệu mật mã về việc tạo / tác giả. |
| NGƯỜI YÊU | Dòng máy dò | SOTA chống giả mạo dựa trên đồ thị attention. |

## Đọc thêm

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — benchmark hiện tại.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — mặc định hình mờ.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE phát hiện cho các cuộc tấn công thời gian.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — đường trục phát hiện SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) - đánh giá độ bền bỉ.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — định dạng bản kê khai xuất xứ.
