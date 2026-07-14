# Chuyển văn bản thành giọng nói (TTS) — Từ Tacotron đến F5 và Kokoro

> ASR đảo ngược lời nói thành văn bản; TTS đảo ngược văn bản thành giọng nói. stack 2026 gồm ba phần: → tokens văn bản, tokens → mel, mel → dạng sóng. Mỗi bộ phận có một model mặc định phù hợp với máy tính xách tay.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ & Mel), Giai đoạn 5 · 09 (Seq2Seq), Giai đoạn 7 · 05 (Full Transformer)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn có một sợi dây: "Xin hãy nhắc tôi tưới cây lúc 6 giờ chiều." Bạn cần một đoạn âm thanh dài 3 giây nghe tự nhiên, có ngữ điệu chính xác (tạm dừng, trọng âm), phát âm "cây" với nguyên âm phù hợp và chạy dưới 300 mili giây trên CPU cho trợ lý giọng nói trực tiếp. Bạn cũng cần hoán đổi giọng nói, xử lý đầu vào chuyển đổi mã ("nhắc tôi lúc 6 giờ chiều, daijoubu?"), và không xấu hổ về tên.

TTS pipelines hiện đại trông như thế này:

1. **Giao diện người dùng văn bản.** Chuẩn hóa văn bản (ngày, số, email), chuyển đổi thành âm vị hoặc tokens phụ, dự đoán features ngữ điệu.
2. **Acoustic model.** Quang phổ văn bản → mel. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → dạng sóng. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), bộ mã hóa codec thần kinh vào năm 2024+.

Vào năm 2026, bộ mã hóa âm thanh + vocoder bị mờ với models khuếch tán từ đầu đến cuối và khớp dòng chảy. Nhưng model tinh thần của ba phần vẫn còn để gỡ lỗi.

## Khái niệm

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: ký tự embedding → BiLSTM encoder → LSTM nhạy cảm vị trí attention → tự hồi quy decoder phát ra khung mel. Chậm (AR), lung lay trên văn bản dài. Vẫn được trích dẫn làm cơ sở.

**FastSpeech 2 (2020).** Không tự hồi quy. Công cụ dự đoán thời lượng xuất ra số lượng khung mel mà mỗi âm vị nhận được. 1 lượt, nhanh hơn 10× so với Tacotron. Mất đi một số tính tự nhiên (alignment đơn điệu) nhưng ships ở khắp mọi nơi.

**VITS (2021).** Cùng huấn luyện encoder + thời lượng dựa trên dòng chảy + bộ mã hóa HiFi-GAN từ đầu đến cuối với inference biến thể. Chất lượng cao, model đơn. Mã nguồn mở thống trị TTS 2022–2024. Các biến thể: YourTTS (zero-shot nhiều loa), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Khuếch tán transformer kết hợp dòng chảy. Prosody tự nhiên, nhân bản giọng nói zero-shot với âm thanh tham chiếu 5 giây. Đứng đầu bảng xếp hạng TTS mã nguồn mở năm 2026. 335 triệu tham số.

**Kokoro (2024).** Nhỏ (82 triệu), có thể chạy CPU, TTS tiếng Anh tốt nhất trong class để sử dụng trong thời gian thực. Từ vựng đóng chỉ tiếng Anh, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Thương mại hiện đại. Thẻ cảm xúc ElevenLabs v2.5 ("[thì thầm]", "[cười]") và giọng nói của nhân vật thống trị production sách nói vào năm 2026.

### Sự phát triển của Vocoder

| Kỷ nguyên | Bộ mã hóa | Độ trễ | Chất lượng |
|-----|---------|---------|---------|
| 2016 | Mạng sóng | Chỉ ngoại tuyến | SOTA khi phát hành |
| 2018 | SóngRNN | ~Thời gian thực | tốt |
| 2020 | HiFi-GAN | 100× thời gian thực | gần người |
| 2022 | BigVGAN | 50× thời gian thực | khái quát hóa trên speakers/langs |
| 2024 | SNAC, DAC (codec thần kinh) | tích hợp với AR models | tokens rời rạc, hiệu quả bit |

Đến năm 2026, hầu hết các models "TTS" đều từ đầu đến cuối từ văn bản sang dạng sóng; Quang phổ Mel là một biểu diễn bên trong.

### Đánh giá

- **MOS (Điểm ý kiến trung bình).** Thang điểm 1–5, có nguồn gốc từ cộng đồng. Vẫn là bản vị vàng; chậm một cách đau đớn.
- **CMOS (MOS so sánh).** Tùy chọn A-vs-B. Khoảng tin cậy chặt chẽ hơn cho mỗi chú thích.
- **UTMOS, DNSMOS.** Trình dự đoán MOS thần kinh không cần tham chiếu. Được sử dụng cho bảng xếp hạng.
- **CER (Tỷ lệ lỗi ký tự) qua ASR.** Chạy TTS đầu ra thông qua Whisper, tính toán CER so với văn bản đầu vào. Proxy để dễ hiểu.
- **SECS (Tương tự Embedding Cosin của loa).** Chất lượng sao chép giọng nói.

Số liệu năm 2026 trên LibriTTS test-clean:

| Model | UTMOS | CER (thông qua Whisper) | Kích thước |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335 triệu |
| XTTS phiên bản 2 | 3.81 | 3.5% | 470 triệu |
| VITS | 3.62 | 3.1% | 25 triệu |
| Kokoro v0.19 | 3.87 | 1.8% | 82 triệu |
| Parler-TTS Lớn | 3.76 | 2.8% | 2,3 tỷ |

## Tự xây dựng

### Bước 1: đầu vào âm vị

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Âm vị là cầu nối phổ quát. Tránh cung cấp văn bản thô cho bất kỳ thứ gì dưới chất lượng cấp VITS.

### Bước 2: chạy Kokoro (2026 CPU mặc định)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Chạy ngoại tuyến, một tệp, 82 triệu tham số.

### Bước 3: chạy F5-TTS với nhân bản giọng nói

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Chuyển clip tham chiếu dài 5 giây + bản ghi của nó; F5 nhân bản prosody và âm sắc.

### Bước 4: Bộ mã hóa HiFi-GAN từ đầu

Quá lớn để phù hợp với một script hướng dẫn, nhưng hình dạng là:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Training: đối nghịch (phân biệt trên windows ngắn) + loss tái tạo quang phổ mel + loss khớp feature. Hàng hóa - sử dụng pretrained checkpoints từ `hifi-gan` repo hoặc nvidia-NeMo.

### Bước 5: pipeline đầy đủ (mã giả)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Trợ lý giọng nói tiếng Anh thời gian thực | Kokoro (CPU) hoặc XTTS v2 (GPU) |
| Nhân bản giọng nói từ tham chiếu 5 giây | F5-TTS |
| Giọng nói nhân vật thương mại | Phòng thí nghiệm ElevenLabs phiên bản 2.5 |
| Tường thuật sách nói | ElevenLabs v2.5 hoặc XTTS v2 + fine-tune |
| Ngôn ngữ tài nguyên thấp | Huấn luyện VITS trên dữ liệu mục tiêu-lang 5–20 giờ |
| Thẻ biểu cảm / cảm xúc | ElevenLabs v2.5 hoặc StyleTTS 2 fine-tune |

Dẫn đầu mã nguồn mở tính đến năm 2026: **F5-TTS về chất lượng, Kokoro về hiệu quả**. Đừng tìm đến Tacotron trừ khi bạn là một nhà sử học.

## Cạm bẫy

- **Không có trình chuẩn hóa văn bản.** "Tiến sĩ Smith" đọc là "Bác sĩ" hoặc "Lái xe"? "2026" là "hai mươi hai mươi sáu" hay "hai không hai sáu"? Chuẩn hóa TRƯỚC khi phát âm.
- **OOV danh từ riêng.** "Ghumare" → "ghyu-mair"? Ship một model grapheme-to-phoneme dự phòng cho các tokens không xác định.
- **Clipping.** Đầu ra Vocoder hiếm khi bị clip, nhưng tỷ lệ mel không phù hợp ở inference có thể vượt quá ±1.0. Luôn `np.clip(wav, -1, 1)`.
- **Tốc độ lấy mẫu không khớp.** Kokoro xuất ra 24 kHz; pipeline xuôi dòng của bạn mong đợi 16 kHz → lấy mẫu lại hoặc lấy răng cưa.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-tts-designer.md`. Thiết kế TTS pipeline cho một giọng nói, độ trễ và mục tiêu ngôn ngữ nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xây dựng từ điển âm vị từ từ vựng đồ chơi, ước tính thời lượng cho mỗi âm vị và in lịch trình "mel" giả.
2. **Trung bình.** Cài đặt Kokoro, tổng hợp cùng một câu ở `af_bella` giọng nói và `am_adam`. So sánh thời lượng âm thanh và chất lượng chủ quan.
3. **Khó.** Quay một clip tham khảo dài 5 giây của chính bạn. Sử dụng F5-TTS để sao chép nó. Báo cáo SECS giữa đầu ra tham chiếu và đầu ra sao chép.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Âm vị | Đơn vị âm thanh | class âm thanh trừu tượng; 39 bằng tiếng Anh (ARPABet). |
| Công cụ dự đoán thời lượng | Mỗi âm vị tồn tại bao lâu | Đầu ra model không AR; khung số nguyên trên mỗi âm vị. |
| Bộ mã hóa | Dạng sóng Mel → | Ánh xạ mạng nơ-ron mel-spec đến các mẫu thô. |
| HiFi-GAN | Bộ mã hóa tiêu chuẩn | dựa trên GAN; thống trị 2020–2024. |
| MOS | Chất lượng chủ quan | 1–5 điểm ý kiến trung bình từ những người đánh giá. |
| GIÂY | Chỉ số sao chép giọng nói | Sự tương đồng cosin giữa embedding loa mục tiêu và đầu ra. |
| F5-TTS | SOTA mã nguồn mở 2024 | Khuếch tán phù hợp với dòng chảy; zero-shot nhân bản. |
| Kokoro | CPU lãnh đạo tiếng Anh | 82 triệu tham số model, Apache 2.0. |

## Đọc thêm

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) - đường cơ sở seq2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) - dựa trên luồng đầu cuối.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — SOTA mã nguồn mở hiện tại.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) - bộ mã hóa vẫn còn ships vào năm 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) - TTS tiếng Anh thân thiện với CPU năm 2024.
