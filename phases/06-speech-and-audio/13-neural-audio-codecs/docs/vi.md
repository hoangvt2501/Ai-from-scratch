# Bộ giải mã âm thanh thần kinh - EnCodec, SNAC, Mimi, DAC và phân tách ngữ nghĩa-âm thanh

> Thế hệ âm thanh năm 2026 gần như là tokens. EnCodec, SNAC, Mimi và DAC biến các dạng sóng liên tục thành các chuỗi rời rạc mà transformer có thể dự đoán. Sự phân chia token ngữ nghĩa và âm thanh - sách mã đầu tiên là ngữ nghĩa, rest là âm thanh - là sự thay đổi kiến trúc quan trọng nhất kể từ khi Transformer âm thanh.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ), Giai đoạn 10 · 11 (Quantization), Giai đoạn 5 · 19 (Từ phụ Tokenization)
**Thời lượng:** ~60 phút

## Vấn đề

Ngôn ngữ models hoạt động trên các tokens rời rạc. Âm thanh liên tục. Nếu bạn muốn có một model theo phong cách LLM cho lời nói / âm nhạc - MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus - trước tiên bạn cần một **codec âm thanh thần kinh**: một encoder học được phân biệt âm thanh thành một từ vựng nhỏ của tokens và một decoder phù hợp để tái tạo dạng sóng.

Hai gia đình đã xuất hiện:

1. **Codec ưu tiên tái tạo** — EnCodec, DAC. Tối ưu hóa chất lượng âm thanh tri giác. Tokens là "âm thanh" - chúng ghi lại mọi thứ bao gồm nhận dạng người nói, âm sắc, nhiễu xung quanh.
2. **Codec ưu tiên ngữ nghĩa** — Mimi (Kyutai), SpeechTokenizer. Buộc codebook đầu tiên mã hóa nội dung ngôn ngữ / ngữ âm (thường bằng cách chắt lọc từ WavLM). Các sách mã tiếp theo là chi tiết âm thanh.

Thông tin chi tiết 2024-2026: **codec tái tạo thuần túy cung cấp cho bạn giọng nói mờ khi bạn cố gắng tạo từ văn bản.** LLM qua codec tokens phải học cả cấu trúc ngôn ngữ VÀ cấu trúc âm thanh trong cùng một sách mã, không mở rộng quy mô. Tách chúng - sách mã ngữ nghĩa 0, sách mã âm thanh 1-N - là điều làm cho Moshi và Sesame CSM hoạt động.

## Khái niệm

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Thủ thuật cốt lõi: Vector Quantization dư (RVQ)

Thay vì một cuốn sách mã lớn (cần hàng triệu mã để có chất lượng tốt), tất cả các codec âm thanh hiện đại đều sử dụng **RVQ**: một loạt các sách mã nhỏ. Sách mã đầu tiên lượng tử hóa đầu ra encoder; thứ hai định lượng phần dư; vân vân. Mỗi cuốn sách mã là 1024 mã. 8 sách mật mã = từ vựng hiệu quả của 1024^8 = 10^24.

Tại thời điểm inference, decoder tổng tất cả các mã đã chọn trên mỗi khung hình để tái tạo.

### Bốn codec quan trọng vào năm 2026

**EnCodec (Meta, 2022).** Đường cơ sở. Encoder-decoder qua dạng sóng, tắc nghẽn RVQ. 24 kHz, có thể có 32 sách mã, mặc định 4 sách mã @ 1.5 kbps. Sử dụng kiến trúc `1D conv + transformer + 1D conv`. Được sử dụng bởi MusicGen.

**DAC (Mô tả, 2023).** RVQ với sách mã chuẩn hóa L2, chức năng kích hoạt định kỳ, cải thiện tổn thất. Độ trung thực tái tạo cao nhất so với bất kỳ codec mở nào - đôi khi không thể phân biệt được với giọng nói gốc với 12 sách mã. 44.1 kHz toàn băng tần.

**SNAC (Hubert Siuzdak, 2024).** RVQ đa tỷ lệ — sách mã thô hoạt động ở tốc độ khung hình thấp hơn sách nhỏ. models âm thanh theo thứ bậc một cách hiệu quả: một "bản phác thảo" thô ở ~12 Hz cộng với chi tiết ở 50 Hz. Được Orpheus-3B sử dụng vì cấu trúc phân cấp ánh xạ tốt vào thế hệ dựa trên LM.

**Mimi (Kyutai, 2024).** Người thay đổi cuộc chơi năm 2026. Tốc độ khung hình 12,5 Hz (cực thấp), 8 sách mã @ 4,4 kbps. Codebook 0 được **chắt lọc từ WavLM** - được huấn luyện để dự đoán features nội dung giọng nói của WavLM. Sách mã 1-7 là phần dư âm thanh. Sự phân chia này cung cấp sức mạnh cho Moshi (Bài học 15) và Sesame CSM.

### Tốc độ khung hình quan trọng đối với mô hình ngôn ngữ

Tốc độ khung hình thấp hơn = trình tự ngắn hơn = LM nhanh hơn.

| Bộ giải mã | Tốc độ khung hình | 1 giây = N khung hình | Tốt cho |
|-------|-----------|----------------|---------|
| Bộ mã hóa-24k | 75 giờ | 75 | âm nhạc, âm thanh chung |
| DAC-44.1k | 86 giờ | 86 | Âm nhạc có độ trung thực cao |
| SNAC-24k (thô) | ~12 giờ | 12 | AR-LM hiệu quả |
| Mimi | 12.5 Hz | 12.5 | streaming phát biểu |

Ở tần số 12,5 Hz, lời nói 10 giây chỉ bằng 125 khung hình codec - một transformer có thể dễ dàng dự đoán chúng.

### tokens ngữ nghĩa so với âm thanh

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (sách mật mã 0 trong Mimi).** Mã hóa những gì đã được nói - âm vị, từ, nội dung. Chắt lọc từ WavLM thông qua một dự đoán phụ trợ loss.
- **tokens âm thanh (sách mã 1-7).** Mã hóa âm sắc, nhận dạng loa, prosody, nhiễu xung quanh, chi tiết tốt.

AR LM dự đoán token ngữ nghĩa trước (điều kiện dựa trên văn bản), sau đó dự đoán tokens âm thanh (có điều kiện về ngữ nghĩa + tham chiếu người nói). Sự phân tích này là lý do tại sao TTS hiện đại có thể zero-shot sao chép giọng nói: model ngữ nghĩa xử lý nội dung; model acoustic xử lý âm sắc.

### Chất lượng tái tạo năm 2026 (bit mỗi giây, tốc độ bit thấp hơn sẽ tốt hơn)

| Bộ giải mã | Tốc độ bit | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kb/giây | 4.0 | 4.3 |
| Bộ mã hóa-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kb/giây | 3.3 | 3.8 |
| Mimi-4.4kbps | 4,4 kb/giây | 3.1 | 3.7 |

Các codec truyền thống như Opus vẫn giành chiến thắng về chất lượng nhận thức. Codec thần kinh giành chiến thắng trên **tokens rời rạc **(mà Opus không sản xuất) và **chất lượng model tổng quát** (những gì LM có thể làm với những tokens đó).

## Tự xây dựng

### Bước 1: mã hóa bằng EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` ở tốc độ 6 kbps. Mỗi mã là 0-1023 (10-bit).

### Bước 2: giải mã và đo lường tái tạo

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Bước 3: phân tách ngữ nghĩa-âm thanh (kiểu Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Sách mã ngữ nghĩa 0 được căn chỉnh theo WavLM. Bạn có thể huấn luyện một transformer văn bản thành ngữ nghĩa - vốn từ vựng nhỏ hơn nhiều so với chuyển trực tiếp sang âm thanh. Sau đó, một điều kiện decoder âm thanh thành dạng sóng riêng biệt trên tham chiếu loa.

### Bước 4: tại sao AR LM qua codec tokens hoạt động

Đối với clip bài phát biểu 10 giây ở sách mã 12.5 Hz × 8 của Mimi:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens là một bối cảnh tầm thường đối với một transformer. Một parameter transformer 256M có thể tạo ra 10 giây giọng nói tính bằng mili giây trên một GPU hiện đại.

## Ứng dụng

Bài toán bản đồ → codec:

| Nhiệm vụ | Bộ giải mã |
|------|-------|
| Thế hệ âm nhạc nói chung | Bộ mã hóa-24k |
| Tái tạo độ trung thực cao nhất | DAC-44.1k |
| AR LM qua giọng nói (TTS) | SNAC hoặc Mimi |
| Streaming giọng nói song công | Mimi (12.5 Hz) |
| Thư viện hiệu ứng âm thanh có văn bản | Điều kiện EnCodec + T5 |
| Chỉnh sửa âm thanh chi tiết | DAC + sơn |

Quy tắc chung: **nếu bạn đang xây dựng một model tổng quát, hãy bắt đầu với Mimi hoặc SNAC. Nếu bạn đang xây dựng một pipeline nén, hãy sử dụng Opus.**

## Cạm bẫy

- **Quá nhiều sách mã.** Thêm sách mã làm tăng độ trung thực tuyến tính nhưng độ dài chuỗi LM cũng tuyến tính. Dừng lại ở 8-12.
- **Tốc độ khung hình không khớp.** Training LM trên 12,5 Hz Mimi sau đó fine-tuning trên 50 Hz EnCodec không thành công.
- **Giả sử tất cả các sách mã bằng nhau.** Trong Mimi, sách mã 0 mang nội dung; Mất nó sẽ phá hủy khả năng hiểu biết. Mất codebook 7 hầu như không đáng chú ý.
- **Sử dụng chất lượng tái tạo làm thước đo duy nhất.** Một codec có thể có khả năng tái tạo tuyệt vời nhưng vô dụng đối với việc tạo dựa trên LM nếu cấu trúc ngữ nghĩa kém.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-codec-picker.md`. Chọn một codec cho một tác vụ tổng quát hoặc nén nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó thực hiện một bộ định lượng vô hướng + dư đồ chơi và đo lường lỗi tái tạo khi bạn thêm sách mã.
2. **Trung bình.** Cài đặt `encodec` và so sánh 1, 4, 8, 32 sách mã trên một clip bài phát biểu được giữ lại. Vẽ PESQ hoặc MSE so với tốc độ bit.
3. **Khó.** Tải Mimi. Mã hóa clip. Thay thế sổ mã 0 bằng các số nguyên ngẫu nhiên; giải mã. Sau đó thay thế codebook 7 tương tự. So sánh hai tham nhũng - tham nhũng sách mã 0 sẽ phá hủy khả năng hiểu biết; Codebook 7 tham nhũng hầu như không thay đổi bất cứ điều gì.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| RVQ | quantization còn lại | Xếp tầng của các cuốn sách mật mã nhỏ; mỗi lượng tử số dư trước đó. |
| Tốc độ khung hình | Tốc độ codec | Có bao nhiêu token khung hình mỗi giây. Thấp hơn = LM nhanh hơn. |
| Sách mã ngữ nghĩa | Sách mật mã 0 (Mimi) | Codebook chắt lọc từ SSL features; mã hóa nội dung. |
| Sách mã âm thanh | Mọi thứ khác | Âm sắc, prosody, nhiễu, chi tiết tốt. |
| PESQ / ViSQOL | Chất lượng tri giác | Các chỉ số khách quan tương quan với MOS. |
| Bộ mã hóa | Codec meta | Đường cơ sở RVQ; được sử dụng bởi MusicGen. |
| Mimi | Bộ giải mã Kyutai | Tốc độ khung hình 12,5 Hz; phân chia ngữ nghĩa-âm thanh; sức mạnh của Moshi. |

## Đọc thêm

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) - đường cơ sở RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) - mở có độ trung thực cao nhất.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) - RVQ đa quy mô.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) - phân tách ngữ nghĩa-âm thanh, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) - mô hình semantic/acoustic hai giai đoạn.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec RVQ có thể phát trực tuyến ban đầu.
