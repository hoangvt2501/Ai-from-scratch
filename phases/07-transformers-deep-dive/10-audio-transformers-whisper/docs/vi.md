# Transformers âm thanh - Whisper Architecture

> Âm thanh là hình ảnh của tần số theo thời gian. Whisper là một ViT ăn quang phổ mel và nói lại.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 7 · 08 (Encoder-Decoder), Giai đoạn 7 · 09 (ViT)
**Thời lượng:** ~45 phút

## Vấn đề

Trước Whisper (OpenAI, Radford et al. 2022), nhận dạng giọng nói tự động hiện đại (ASR) có nghĩa là wav2vec 2.0 và HuBERT - bộ trích xuất feature tự giám sát cộng với đầu fine-tuned. Chất lượng cao, pipelines dữ liệu đắt tiền, dễ gãy miền. Nhận dạng giọng nói đa ngôn ngữ cần có models riêng biệt cho mỗi nhóm ngôn ngữ.

Whisper đã đặt cược ba lần:

1. **Huấn luyện mọi thứ.** 680.000 giờ âm thanh được dán nhãn yếu được thu thập từ internet trên 97 ngôn ngữ. Không có kho học thuật sạch. Không có nhãn âm vị.
2. **model đơn đa nhiệm.** Một decoder được huấn luyện chung về phiên âm, dịch, phát hiện hoạt động giọng nói, ID ngôn ngữ và đánh dấu thời gian thông qua tokens tác vụ.
3. **encoder-decoder transformer tiêu chuẩn.** Encoder tiêu thụ quang phổ log-mel. Decoder tạo ra văn bản tokens tự hồi quy. Không có vocoder, không CTC, không HMM.

Kết quả: Whisper large-v3 mạnh mẽ trên các trọng âm, nhiễu và ngôn ngữ không có dữ liệu được gắn nhãn rõ ràng. Đây là giao diện người dùng giọng nói mặc định cho mọi trợ lý giọng nói mã nguồn mở và hầu hết các trợ lý thương mại vào năm 2026.

## Khái niệm

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Bước 1 - lấy mẫu lại + cửa sổ

Âm thanh ở 16 kHz. Clip/pad đến 30 giây. Tính toán quang phổ log-mel: 80 thùng mel, sải chân 10 ms → ~3.000 khung hình × 80 features. Đây là "hình ảnh đầu vào" mà Whisper nhìn thấy.

### Bước 2 - thân tích chập

Hai layer Conv1D với kernel 3 và stride 2 giảm 3.000 khung hình xuống còn 1.500. Giảm một nửa độ dài trình tự mà không thêm nhiều parameters.

### Bước 3 - encoder

24 lớp (đối với lớn) transformer encoder hơn 1.500 bước thời gian. Mã hóa vị trí hình sin, self-attention GELU FFN. Tạo ra 1.500 × 1.280 trạng thái ẩn.

### Bước 4 - decoder

Một transformer decoder 24 lớp. Nó tự hồi quy tạo ra tokens từ một từ vựng BPE là một tập hợp siêu GPT-2 với một vài tokens đặc biệt dành riêng cho âm thanh.

### Bước 5 - tokens nhiệm vụ

decoder prompt bắt đầu với tokens kiểm soát cho người model biết phải làm gì:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

hoặc

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

model đã được huấn luyện về quy ước này. Bạn kiểm soát nhiệm vụ bằng tiền tố. Tương đương với điều chỉnh hướng dẫn vào năm 2026, nhưng được áp dụng cho giọng nói.

### Bước 6 - đầu ra

Beam search (chiều rộng 5) với ngưỡng log-prob. Dấu thời gian được dự đoán sau mỗi 0,02 giây âm thanh khi không có `<|notimestamps|>` token.

### Kích thước thì thầm

| Model | Tham số | Lớp | d_model | Đầu | VRAM (FP16) |
|-------|--------|--------|---------|-------|-------------|
| Tí hon | 39 triệu | 4 | 384 | 6 | ~1 GB |
| Căn cứ | 74 triệu | 6 | 512 | 8 | ~1 GB |
| Nhỏ | 244 triệu | 12 | 768 | 12 | ~2 GB |
| Trung bình | 769 triệu | 24 | 1024 | 16 | ~5 GB |
| Lớn | 1550 triệu | 32 | 1280 | 20 | ~10 GB |
| Lớn-v3 | 1550 triệu | 32 | 1280 | 20 | ~10 GB |
| V3-turbo lớn | 809 triệu | 32 | 1280 | 20 | ~6 GB (decoder 4 lớp) |

Large-v3-turbo (2024) cắt giảm decoder từ 32 lớp xuống còn 4 lớp. Giải mã nhanh hơn 8× với hồi quy điểm WER <1. Mở khóa tốc độ giải mã đó là lý do tại sao Whisper-turbo là mặc định cho agents thoại thời gian thực vào năm 2026.

### Những gì Whisper không làm

- Không diaration (ai đang nói). Ghép nối với pyannote cho điều đó.
- Không có streaming thời gian thực - cửa sổ 30 giây đã được cố định. Bao bì hiện đại (`faster-whisper`, `WhisperX`) bắt vít streaming thông qua VAD + chồng chéo.
- Không có bối cảnh dạng dài nào vượt quá 30 giây mà không có phân đoạn bên ngoài. Hoạt động tốt trong thực tế vì lời nói của con người hiếm khi cần ngữ cảnh tầm xa để phiên âm.

### Phong cảnh năm 2026

| Nhiệm vụ | Model | Ghi chú |
|------|-------|-------|
| Tiếng Anh ASR | Thì thầm-turbo, Moonshine | Moonshine nhanh hơn 4× trên cạnh |
| ASR đa ngôn ngữ | Thì thầm-lớn-v3 | 97 ngôn ngữ |
| Streaming ASR | thì thầm nhanh hơn + VAD | Mục tiêu độ trễ 150 ms có thể đạt được |
| TTS | Piper, XTTS-v2, Kokoro | Encoder decoder mẫu, nhưng có hình Whisper |
| Âm thanh + ngôn ngữ | AudioLM, liền mạchM4T | tokens văn bản + tokens âm thanh trong một transformer |

## Tự xây dựng

Xem `code/main.py`. Chúng ta không huấn luyện Whisper - chúng tôi xây dựng bộ định dạng token prompt pipeline quang phổ log-mel + task-. Đó là những phần bạn thực sự chạm vào trong production.

### Bước 1: tổng hợp âm thanh

Tạo sóng sin 1 giây ở 440 Hz được lấy mẫu ở 16 kHz. 16.000 mẫu.

### Bước 2: quang phổ log-mel (đơn giản hóa)

Quang phổ mel đầy đủ cần FFT. Chúng ta thực hiện một phiên bản tạo khung hình đơn giản + năng lượng trên mỗi khung hình hiển thị pipeline mà không yêu cầu `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Khung = 25 ms, bước nhảy = 10 ms. Khớp với cửa sổ của Whisper. Năng lượng trên mỗi khung là viết tắt của mel bins cho sư phạm.

### Bước 3: đệm đến 30 giây

Whisper luôn processes các đoạn 30 giây. Đệm (hoặc cắt) quang phổ thành 3.000 khung hình.

### Bước 4: xây dựng prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Đó là toàn bộ bề mặt kiểm soát tác vụ. Tiền tố 4 token.

## Ứng dụng

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Nhanh hơn, tương thích với OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Khi nào nên chọn Whisper vào năm 2026:**

- ASR đa ngôn ngữ với một model.
- Phiên âm mạnh mẽ của âm thanh nhiễu, đa dạng.
- Nghiên cứu / nguyên mẫu ASR - điểm khởi đầu nhanh nhất.

**Khi nào nên chọn thứ khác:**

- Độ trễ cực thấp streaming cạnh — Moonshine đánh bại Whisper với chất lượng phù hợp.
- AI đàm thoại thời gian thực cần <200 ms - streaming ASR chuyên dụng.
- Diarization của loa - Whisper không làm điều này; bu lông trên pyannote.

## Sản phẩm bàn giao

Xem `outputs/skill-asr-configurator.md`. skill chọn một ASR model, giải mã parameters và tiền xử lý pipeline cho một ứng dụng giọng nói mới.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác nhận số khung hình cho tín hiệu 1 giây ở 16 kHz với bước nhảy 10 ms là ~100 khung hình. Trong 30 giây: ~3.000 khung hình.
2. **Trung bình.** Xây dựng quang phổ log-mel đầy đủ bằng cách sử dụng `numpy.fft`. Xác minh 80 thùng mel khớp `librosa.feature.melspectrogram(n_mels=80)` trong lỗi số.
3. **Khó.** Triển khai streaming inference: chia đoạn âm thanh thành 10 giây windows với 2 giây chồng chéo, chạy Whisper trên mỗi đoạn merge bản ghi. Đo tỷ lệ lỗi từ so với một lần qua trên mẫu podcast dài 5 phút.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Quang phổ Mel | "Hình ảnh âm thanh" | Biểu diễn 2D: thùng tần số trên một trục, khung thời gian trên trục kia; năng lượng theo tỷ lệ log trên mỗi tế bào. |
| Khúc gỗ | "Những gì Whisper nhìn thấy" | Quang phổ Mel đi qua nhật ký; xấp xỉ nhận thức của con người về âm lượng. |
| Khung | "Một lát một lần" | Cửa sổ mẫu 25 ms; chồng lên nhau ở sải chân 10 ms. |
| Nhiệm vụ token | "Prompt tiền tố cho lời nói" | Các tokens đặc biệt như '<\ | Phiên âm\ | >` / `<\ | dịch\ | >' trong decoder prompt. |
| Phát hiện hoạt động giọng nói (VAD) | "Tìm bài phát biểu" | Cổng xóa bỏ sự im lặng trước khi ASR; cắt giảm chi phí khổng lồ. |
| CTC | "Phân loại thời gian kết nối" | ASR loss cổ điển cho training không alignment; Whisper KHÔNG sử dụng nó. |
| Thì thầm-turbo | "decoder nhỏ, đầy encoder" | encoder lớn v3 + decoder 4 lớp; 8× Giải mã nhanh hơn. |
| Thì thầm nhanh hơn | "Giấy gói production" | Triển khai lại CTranslate2; int8 quantization; 4× nhanh hơn tài liệu tham khảo của OpenAI. |

## Đọc thêm

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - Giấy thì thầm.
- [OpenAI Whisper repo](https://github.com/openai/whisper) - mã tham chiếu + model trọng lượng. Đọc `whisper/model.py` để xem thân Conv1D + encoder + decoder từ trên xuống dưới trong ~400 dòng.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) - logic tìm kiếm chùm tia + token tác vụ được mô tả trong Bước 5–6 ở đây; 500 dòng, có thể đọc đầy đủ.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — tiền chất; vẫn features SOTA trong một số cài đặt.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) - production bao bọc, nhanh hơn 4× so với tham chiếu.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) - ASR thân thiện với cạnh năm 2024, hình Whisper nhưng nhỏ hơn.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) - công thức fine-tuning chuẩn bao gồm bộ tiền xử lý quang phổ mel và xử lý dấu thời gian token.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — triển khai đầy đủ (encoder, decoder, cross-attention, thế hệ) phản ánh sơ đồ kiến trúc của bài học.
