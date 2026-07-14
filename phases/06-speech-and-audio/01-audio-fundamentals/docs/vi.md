# Các nguyên tắc cơ bản về âm thanh - Dạng sóng, Sampling, Biến đổi Fourier

> Dạng sóng là tín hiệu thô. Quang phổ là biểu diễn. Mel features là hình thức thân thiện với ML. Mọi ASR và TTS pipeline hiện đại đều đi trên nấc thang này, và nấc thang đầu tiên là hiểu sampling và Fourier.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 1 · 06 (Vectors & Ma trận), Giai đoạn 1 · 14 (Phân phối xác suất)
**Thời lượng:** ~45 phút

## Vấn đề

Micrô tạo ra tín hiệu áp suất so với thời gian. Mạng nơ-ron của bạn tiêu thụ tensors. Giữa chúng là một stack quy ước, khi bị vi phạm, tạo ra lỗi im lặng: model huấn luyện tốt nhưng WER tăng gấp đôi, hoặc TTS ships tiếng rít, hoặc hệ thống nhân bản giọng nói ghi nhớ micrô thay vì loa.

Mọi lỗi trong hệ thống giọng nói đều traces trở lại một trong ba câu hỏi:

1. Dữ liệu được ghi lại ở tốc độ lấy mẫu là bao nhiêu và model mong đợi điều gì?
2. Tín hiệu có bí danh không?
3. Bạn đang hoạt động trên các mẫu thô hay trên biểu diễn tần số?

Hãy làm đúng những điều này và rest của Giai đoạn 6 có thể giải quyết được. Hiểu sai chúng và ngay cả Whisper-Large-v4 cũng tạo ra rác.

## Khái niệm

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Dạng sóng.** Một mảng nổi một chiều trong `[-1.0, 1.0]`. Lập chỉ mục theo số mẫu. Để chuyển đổi sang giây, chia cho tốc độ lấy mẫu: `t = n / sr`. Một clip dài 10 giây ở tần số 16 kHz là một mảng gồm 160.000 phao.

**Sampling tốc độ (sr).** Có bao nhiêu mẫu mỗi giây. Tỷ lệ phổ biến vào năm 2026:

| Tỷ lệ | Sử dụng |
|------|-----|
| 8 kHz | Điện thoại, VOIP kế thừa. Nyquist ở 4 kHz giết chết phụ âm. Tránh cho ASR. |
| 16 kHz | ASR tiêu chuẩn. Whisper, Parakeet, SeamlessM4T v2 đều tiêu thụ 16 kHz. |
| 22,05 kHz | TTS vocoder training dành cho models lớn tuổi. |
| 24 kHz | TTS hiện đại (Kokoro, F5-TTS, xTTS v2). |
| 44,1 kHz | CD âm thanh, âm nhạc. |
| 48 kHz | Phim, âm thanh chuyên nghiệp, TTS độ trung thực cao (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Tốc độ lấy mẫu `sr` có thể biểu diễn rõ ràng các tần số lên đến `sr/2`. Ranh giới `sr/2` là *Nyquist frequency*. Năng lượng phía trên Nyquist được *bí danh* - gấp lại thành các tần số thấp hơn - và làm hỏng tín hiệu. Luôn luôn bộ lọc thông thấp trước khi xuốngampling.

**Độ sâu bit.** PCM 16-bit (int16 có ký, phạm vi ±32.767) là định dạng trao đổi phổ quát. 24-bit cho âm nhạc, 32-bit float cho DSP nội bộ. Các thư viện như `soundfile` đọc int16 nhưng hiển thị các mảng float32 trong `[-1, 1]`.

**Biến đổi Fourier.** Bất kỳ tín hiệu hữu hạn nào là tổng của các hình sin ở các tần số khác nhau. Biến đổi Fourier rời rạc (DFT) tính toán, đối với `N` mẫu, `N` hệ số phức - một hệ số trên mỗi thùng tần số. `bin k` ánh xạ đến tần số `k · sr / N` Hz. Độ lớn là biên độ ở tần số đó, góc là pha.

**FFT.** Biến đổi Fourier nhanh: một thuật toán `O(N log N)` cho DFT khi `N` là lũy thừa của 2. Mọi thư viện âm thanh đều sử dụng FFT dưới mui xe. FFT 1024 mẫu ở 16 kHz cung cấp 512 thùng tần số có thể sử dụng kéo dài 0–8 kHz ở độ phân giải 15.6 Hz.

**Khung hình + cửa sổ.** Chúng ta không FFT toàn bộ clip. Chúng ta cắt nó thành các * khung hình * chồng lên nhau (thường là 25 ms với bước nhảy 10 ms), nhân mỗi khung hình với một hàm cửa sổ (Hann, Hamming) để loại bỏ sự gián đoạn cạnh, sau đó FFT mỗi khung hình. Đây là Biến đổi Fourier thời gian ngắn (STFT). Bài 02 tiếp tục từ đây.

```figure
mel-scale
```

## Tự xây dựng

### Bước 1: đọc clip và vẽ dạng sóng

`code/main.py` chỉ sử dụng mô-đun `wave` stdlib để giữ cho bản demo không phụ thuộc. Đối với production bạn sẽ sử dụng `soundfile` hoặc `torchaudio.load` (cả hai đều trả về `(waveform, sr)` bộ dữ liệu):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Bước 2: tổng hợp sóng sin từ các nguyên lý đầu tiên

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Một hình sin 440 Hz (concert A) ở 16 kHz trong 1 giây là 16.000 float. Ghi bằng `wave.open(..., "wb")` bằng mã hóa PCM 16-bit.

### Bước 3: tính DFT bằng tay

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` - tốt cho `N=256` xác nhận tính chính xác, vô dụng đối với âm thanh thực. Lệnh gọi mã thực `numpy.fft.rfft` hoặc `torch.fft.rfft`.

### Bước 4: tìm tần số thống trị

Chỉ số cường độ cực đại `k_star` ánh xạ đến tần số `k_star * sr / N`. Chạy điều này trên sin 440 Hz sẽ trả về đỉnh ở thùng `440 * N / sr`.

### Bước 5: minh họa răng cưa

Lấy mẫu một hình sin 7 kHz ở 10 kHz (Nyquist = 5 kHz). Âm 7 kHz nằm trên Nyquist và gập lại để `10 − 7 = 3 kHz`. Đỉnh FFT xuất hiện ở 3 kHz. Đây là bản demo răng cưa cổ điển và lý do mọi DAC/ADC ships với bộ lọc thông thấp trên tường gạch.

## Ứng dụng

stack bạn sẽ thực sự ship vào năm 2026:

| Nhiệm vụ | Thư viện | Tại sao |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (trình bao bọc libsndfile) | Nhanh nhất, ổn định, trả về float32. |
| Lấy mẫu lại | `torchaudio.transforms.Resample` hoặc `librosa.resample` | Khử răng cưa chính xác được tích hợp sẵn. |
| STFT / Mel | `torchaudio` hoặc `librosa` | thân thiện với GPU; PyTorch hệ sinh thái. |
| streaming thời gian thực | `sounddevice` hoặc `pyaudio` | Liên kết PortAudio đa nền tảng. |
| Kiểm tra tệp | `ffprobe` hoặc `soxi` | CLI, nhanh chóng, báo cáo sr/channels/codec. |

Quy tắc quyết định: **khớp với tốc độ lấy mẫu trước khi bạn khớp với bất kỳ thứ gì khác**. Whisper mong đợi phao đơn âm 16 kHz32. Chuyển nó âm thanh nổi 44.1 kHz và bạn sẽ nhận được rác trông giống như một lỗi model.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-audio-loader.md`. Công skill giúp bạn kiểm tra xem đầu vào âm thanh có phù hợp với mong đợi của model xuôi dòng hay không và lấy mẫu lại một cách chính xác khi không.

## Bài tập

1. **Dễ dàng.** Tổng hợp hỗn hợp 1 giây gồm 220 Hz + 440 Hz + 880 Hz ở 16 kHz. Chạy DFT. Xác nhận ba đỉnh tại các thùng dự kiến.
2. **Trung bình.** Ghi lại WAV 3 giây của giọng nói của bạn ở 48 kHz. Downsample xuống 16 kHz bằng cách sử dụng `torchaudio.transforms.Resample` (có khử răng cưa), sau đó lên 16 kHz bằng cách sử dụng phân rã ngây thơ (mỗi mẫu thứ ba). FFT cả hai. Răng cưa xuất hiện ở đâu?
3. **Khó.** Xây dựng STFT từ đầu chỉ bằng cách sử dụng `math` và DFT từ Bước 3. Kích thước khung 400, nhảy 160, cửa sổ Hann. Vẽ quy mô với `matplotlib.pyplot.imshow`. Đây là quang phổ của Bài 02.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Tỷ lệ lấy mẫu | Có bao nhiêu mẫu mỗi giây | Tần số tính bằng Hz mà tại đó ADC đo tín hiệu. |
| Nyquist | Tần số tối đa bạn có thể đại diện | `sr/2`; năng lượng phía trên nó bí danh trở lại. |
| Độ sâu bit | Độ phân giải của từng mẫu | `int16` = 65.536 mức; `float32` = precision 24 bit trong `[-1, 1]`. |
| ĐTM | Biến đổi Fourier cho dãy | `N` mẫu → `N` hệ số tần số phức tạp. |
| FFT | DFT nhanh | `O(N log N)` thuật toán yêu cầu `N` = lũy thừa của 2. |
| Thùng rác | Cột tần số | `k · sr / N` Hz; độ phân giải = `sr / N`. |
| STFT | Quang phổ dưới mui xe | FFT có khung + cửa sổ theo thời gian. |
| Răng cưa | Bóng ma tần số kỳ lạ | Năng lượng trên Nyquist phản chiếu xuống các thùng thấp hơn. |

## Đọc thêm

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - bài báo đằng sau định lý sampling.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) - sách giáo khoa DSP chuẩn miễn phí.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — hướng dẫn thực tế với mã.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) - tài liệu tham khảo về lý do tại sao âm thanh trong thế giới thực không phải là hình sin sạch.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) - trực giác thùng tần số được xóa sau 10 phút.
