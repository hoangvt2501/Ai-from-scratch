# Quang phổ, Thang đo Mel và Features âm thanh

> Mạng nơ-ron không tiêu thụ tốt dạng sóng thô. Họ tiêu thụ quang phổ. Họ tiêu thụ quang phổ mel thậm chí còn tốt hơn. Mọi ASR, TTS và bộ phân loại âm thanh vào năm 2026 đều sống hoặc chết bởi lựa chọn tiền xử lý duy nhất này.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 01 (Kiến thức cơ bản về âm thanh)
**Thời lượng:** ~45 phút

## Vấn đề

Quay clip 16 kHz dài 10 giây. Đó là 160.000 chiếc phao, tất cả `[-1, 1]`, gần như hoàn toàn không tương quan với nhãn "chó sủa" hoặc "từ mèo". Dạng sóng thô có thông tin nhưng ở dạng model không thể dễ dàng trích xuất. Hai âm vị giống hệt nhau được nói cách nhau 100 ms có các mẫu thô hoàn toàn khác nhau.

Một quang phổ khắc phục điều này. Nó làm sụp đổ chi tiết thời gian nơi nhận thức của con người bỏ qua nó (jitter micro giây) và bảo tồn cấu trúc nơi nhận thức tham dự (tần số có năng lượng, theo thời gian windows ~10–25 ms).

Quang phổ Mel đẩy xa hơn. Con người cảm nhận cao độ theo logarit: 100 Hz so với 200 Hz nghe có vẻ "cách nhau cùng một khoảng cách" với 1000 Hz so với 2000 Hz. Thang mel làm cong trục tần số để phù hợp. Quang phổ tỷ lệ mel là feature quan trọng nhất trong ML giọng nói từ năm 2010 đến năm 2026.

## Khái niệm

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Biến đổi Fourier thời gian ngắn).** Cắt dạng sóng thành các khung chồng lên nhau (điển hình: cửa sổ 25 ms, bước nhảy 10 ms = 400 mẫu / 160 mẫu ở 16 kHz). Nhân mỗi khung với một hàm cửa sổ (Hann là mặc định; Hamming đánh đổi hơi khác một chút). FFT mỗi khung hình. Stack quang phổ cường độ thành một ma trận hình dạng `(n_frames, n_freq_bins)`. Đó là quang phổ của bạn.

**Log-magnitude.** Độ lớn thô span 5-6 bậc độ lớn. Lấy `log(|X| + 1e-6)` hoặc `20 * log10(|X|)` để nén dải động. Mỗi production pipeline sử dụng cường độ log, không phải cấp sao thô.

**Thang Mel.** Tần số `f` tính bằng Hz ánh xạ đến mel `m` theo `m = 2595 * log10(1 + f / 700)`. Ánh xạ gần tuyến tính dưới 1 kHz và gần như logarit ở trên. 80 thùng mel bao phủ 0–8 kHz là đầu vào ASR tiêu chuẩn.

**Mel filterbank.** Một bộ lọc hình tam giác cách đều nhau trên thang mel. Mỗi bộ lọc là tổng trọng số của các thùng FFT liền kề. Nhân độ lớn STFT với ma trận ngân hàng bộ lọc cho quang phổ mel trong một matmul.

**Quang phổ Log-mel.** `log(mel_spec + 1e-10)`. Đầu vào của Whisper. Đầu vào của Vẹt đuôi dài. Đầu vào của SeamlessM4T. Giao diện người dùng âm thanh 2026 phổ quát.

**MFCC.** Lấy quang phổ log-mel, áp dụng DCT (loại II), giữ nguyên 13 hệ số đầu tiên. Khử trùng features và nén thêm. Chiếm ưu thế feature cho đến khoảng năm 2015 khi CNNs/Transformers trên khúc gỗ thô bắt kịp. Vẫn được sử dụng trong nhận dạng loa (x-vectors, ECAPA).

**Giao dịch độ phân giải.** FFT lớn hơn = độ phân giải tần số tốt hơn nhưng độ phân giải thời gian kém hơn. 25 ms / 10 ms là mặc định ML âm thanh; 50 ms / 12,5 ms cho âm nhạc; 5 ms / 2 ms để phát hiện thoáng qua (tiếng trống, tiếng nổ).

```figure
spectrogram-window
```

## Tự xây dựng

### Bước 1: tạo khung dạng sóng

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Clip 10 kHz 16 giây với `frame_len=400, hop=160` mang lại 998 khung hình.

### Bước 2: Cửa sổ Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Nhân phần tử khôn ngoan trước FFT. Loại bỏ rò rỉ quang phổ do cắt bớt ở endpoints không bằng không.

### Bước 3: Độ lớn STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Production sử dụng `torch.stft` hoặc `librosa.stft` (FFT-backed, vectorized). Vòng lặp ở đây là sư phạm; Nó chạy trên các clip ngắn trong `code/main.py`.

### Bước 4: mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel bao phủ 0–8 kHz với `n_fft=400` cho ma trận `(80, 201)`. Nhân độ lớn STFT `(n_frames, 201)` với chuyển vị để có được quang phổ mel `(n_frames, 80)`.

### Bước 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Các lựa chọn thay thế phổ biến: `librosa.power_to_db` (dB chuẩn hóa tham chiếu), `10 * log10(power + eps)`. Whisper sử dụng một clip phức tạp hơn + quy trình chuẩn hóa (xem `log_mel_spectrogram` của Whisper).

### Bước 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Áp dụng DCT cho từng khung log-mel, giữ nguyên 13 hệ số đầu tiên. Đó là ma trận MFCC của bạn. Hệ số đầu tiên thường bị loại bỏ (nó mã hóa năng lượng tổng thể).

## Ứng dụng

stack năm 2026:

| Nhiệm vụ | Features |
|------|----------|
| ASR (Thì thầm, Vẹt đuôi dài, liền mạchM4T) | 80 log-mels, 10 ms hop, cửa sổ 25 ms |
| TTS model âm thanh (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop để kiểm soát thời gian tốt |
| Phân loại âm thanh (AST, PANN, BEATs) | 128 khúc gỗ, 10 ms hop |
| embedding loa (ECAPA-TDNN, WavLM) | 80 log-mels hoặc SSL dạng sóng thô |
| Âm nhạc (MusicGen, Âm thanh ổn định 2) | tokens rời rạc EnCodec (không phải mels) |
| Phát hiện từ khóa | 40 MFCC cho các thiết bị nhỏ |

Quy tắc chung: **nếu bạn không làm việc với âm nhạc, hãy bắt đầu với 80 log-mels.

## Những cạm bẫy vẫn ship vào năm 2026

- **Số lượng mel không khớp.** Training với 80 mel, inference với 128 mel. Thất bại thầm lặng. Ghi nhật ký hình feature ở cả hai đầu.
- **Tốc độ lấy mẫu không khớp ngược dòng.** Mels được tính toán ở 22,05 kHz trông khác với 16 kHz. Sửa SR *trước* tính năng hóa.
- **dB so với log.** Whisper mong đợi log-mel, không phải dB-mel. Một số HF pipelines tự động phát hiện; mã tùy chỉnh của bạn sẽ không.
- **Chuẩn hóa trôi dạt.** Chuẩn hóa mỗi lời nói trong training, chuẩn hóa toàn cầu trong quá trình inference. Production lỗi nhân đôi WER.
- **Rò rỉ từ đệm.** Zero-padding ở cuối clip tạo ra một quang phổ phẳng trong các khung sau. Đệm đối xứng hoặc sao chép.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-feature-extractor.md`. skill chọn loại feature, số lượng mel, frame/hop và chuẩn hóa cho một mục tiêu model nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó tổng hợp một tiếng kêu (tần số quét 200 → 4000 Hz) và in thùng argmax mel trên mỗi khung hình. Vẽ (tùy chọn) và xác nhận nó khớp với quét.
2. **Trung bình.** Chạy lại với `n_mels` trong `{40, 80, 128}` và `frame_len` trong `{200, 400, 800}`. Đo băng thông đỉnh sắc nét trên trục thời gian. Combo nào giải quyết tiếng kêu tốt nhất?
3. **Khó.** Triển khai `power_to_db` và so sánh ASR accuracy của một bộ phân loại CNN nhỏ trên AudioMNIST bằng cách sử dụng (a) log-mel thô, (b) dB-mel với `ref=max`, (c) MFCC-13 + delta + delta-delta. Báo cáo top 1 accuracy.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Khung | Một lát | Khối dạng sóng 25 ms được cung cấp cho một FFT. |
| Bước nhảy | Sải bước | Mẫu giữa các khung liên tiếp; 10 ms là ASR mặc định. |
| Cửa sổ | Hann/Hamming điều | Hệ số nhân theo điểm giúp giảm các cạnh khung hình về không. |
| STFT | Máy tạo quang phổ | FFT có khung + cửa sổ; mang lại ma trận tần số × thời gian. |
| Mel | Tần số bị cong vênh | Thang đo nhận thức nhật ký; `m = 2595·log10(1 + f/700)`. |
| Ngân hàng bộ lọc | Ma trận | Bộ lọc hình tam giác chiếu STFT lên thùng mel. |
| Khúc gỗ | Đầu vào của Whisper | `log(mel_spec + eps)`; được tiêu chuẩn hóa vào năm 2026. |
| MFCC | feature trường học cũ | DCT của log-mel; 13 coeffs, không tương quan. |

## Đọc thêm

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) - bài báo MFCC.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — thang mel ban đầu.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — đọc triển khai tham chiếu.
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — tài liệu tham khảo cho `mfcc`, `melspectrogram` và hop/window.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) - pipeline tỷ lệ production cho Vẹt đuôi dài + Canary models.
