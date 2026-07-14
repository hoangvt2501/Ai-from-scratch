# Xử lý âm thanh thời gian thực

> Batch pipelines process một tệp. Thời gian thực pipelines process 20 mili giây tiếp theo trước khi 20 giây tiếp theo đến. Mọi AI đàm thoại, phòng thu phát sóng và bot điện thoại đều sống và chết bởi ngân sách độ trễ này.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ), Giai đoạn 6 · 04 (ASR), Giai đoạn 6 · 07 (TTS)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn muốn một trợ lý giọng nói cảm thấy sống động. Độ trễ theo lượt đàm thoại của con người là ~230 ms (im lặng để phản hồi). Bất cứ thứ gì trên 500 ms đều có cảm giác như robot; trên 1500 ms cảm thấy bị hỏng. Ngân sách cho một vòng lặp đầy đủ **nghe → hiểu → phản hồi → nói** vào năm 2026 là:

| Sân khấu | Ngân sách |
|-------|--------|
| Bộ đệm → micrô | 20 mili giây |
| VAD | 10 mili giây |
| ASR (streaming) | 150 mili giây |
| LLM (token đầu tiên) | 100 mili giây |
| TTS (đoạn đầu tiên) | 100 mili giây |
| Kết xuất loa → | 20 mili giây |
| **Tổng** | **~400 mili giây **|

Moshi (Kyutai, 2024) đạt tốc độ song công 200 ms. Đồng hồ GPT-4o-realtime (2024) ~320 ms. pipelines xếp tầng vào năm 2022 shipped ở 2500 ms. Cải tiến 10× đến từ ba kỹ thuật: (1) streaming ở khắp mọi nơi, (2) đường ống không đồng bộ với kết quả một phần, (3) tạo gián đoạn.

## Khái niệm

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Khung / khối / cửa sổ.** Âm thanh thời gian thực chảy dưới dạng khối có kích thước cố định. Lựa chọn phổ biến: 20 ms (320 mẫu ở 16 kHz). Mọi thứ ở hạ nguồn phải theo kịp nhịp độ này.

**Bộ đệm vòng.** Bộ đệm tròn kích thước cố định. Producer thread viết các khung hình mới, người tiêu dùng thread đọc. Ngăn chặn phân bổ trong đường dẫn nóng. Kích thước ≈ độ trễ tối đa × tốc độ lấy mẫu; vòng 2 kHz 16 giây = 32.000 mẫu.

**VAD (Phát hiện hoạt động giọng nói).** Cổng xuôi dòng hoạt động khi không có ai nói. Silero VAD 4.0 (2024) chạy <1 ms trên khung hình 30 ms trên CPU. `webrtcvad` là giải pháp thay thế cũ hơn.

**Streaming ASR.** Models phát ra một phần bản ghi khi âm thanh đến. Parakeet-CTC-0.6B ở chế độ streaming (NeMo, 2024) thực hiện 2–5% WER ở độ trễ 320 ms. Whisper-Streaming (Macháček và cộng sự, 2023) chia đoạn Whisper cho gần streaming ở độ trễ ~2 giây.

**Gián đoạn.** Khi người dùng nói trong khi trợ lý đang nói, bạn phải (a) phát hiện sà lan vào, (b) dừng TTS, (c) loại bỏ đầu ra LLM còn lại. Tất cả trong vòng 100 ms hoặc người dùng nhận thấy trợ lý điếc.

**WebRTC Opus transport.** Khung hình 20 ms, 48 kHz, tốc độ bit thích ứng 8–128 kbps. Tiêu chuẩn cho trình duyệt và thiết bị di động. LiveKit, Daily.co, Pion là những stacks năm 2026 để xây dựng ứng dụng thoại.

**Bộ đệm jitter.** Các gói mạng đến không theo thứ tự / muộn. Bộ đệm jitter sắp xếp lại và làm mịn; quá nhỏ → khoảng trống có thể nghe được, độ trễ → quá lớn. 60–80 ms điển hình.

### Các vấn đề phổ biến

- **Thread tranh cãi.** models nặng GIL + của Python có thể làm chết đói thread âm thanh. Sử dụng thư viện âm thanh C-callback (thiết bị âm thanh, PortAudio) và giữ Python tránh xa đường dẫn nóng.
- **Độ trễ chuyển đổi tốc độ lấy mẫu.** Lấy mẫu lại bên trong pipeline thêm 5–20 mili giây. Lấy mẫu lại trước hoặc sử dụng bộ lấy mẫu lại không có độ trễ (PolyPhase, `soxr_hq`).
- **TTS mồi.** Ngay cả TTS nhanh như Kokoro cũng có khả năng khởi động 100–200 ms trong yêu cầu đầu tiên. Cache model + làm ấm nó bằng một lần chạy giả trước lượt thực đầu tiên.
- **Khử tiếng vang.** Nếu không có AEC, TTS ra sẽ đi vào lại micrô và triggers ASR bằng giọng nói của chính bot. WebRTC AEC3 là mặc định mã nguồn mở.

```figure
nyquist-aliasing
```

## Tự xây dựng

### Bước 1: đệm vòng

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Dung lượng xác định độ trễ bộ đệm tối đa. 32.000 mẫu ở 16 kHz = 2 giây.

### Bước 2: Cổng VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Thay thế bằng Silero VAD trong production:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Bước 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Bước 4: Trình xử lý gián đoạn

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Bản lề trên I/O không đồng bộ và TTS streaming có thể hủy bỏ. WebRTC peerconnection.stop() trên bản âm thanh là cách chuẩn.

## Ứng dụng

stack năm 2026:

| Lớp | Chọn |
|-------|------|
| Transport | LiveKit (WebRTC) hoặc Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Vẹt đuôi dài-CTC-0.6B hoặc Whisper-Streaming |
| LLM token đầu tiên | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro hoặc ElevenLabs Turbo v2.5 |
| Hủy tiếng vang | WebRTC AEC3 |
| Gốc từ đầu đến cuối | OpenAI API thời gian thực hoặc Moshi |

## Cạm bẫy

- **Đệm 500 ms để an toàn.** Bộ đệm *là* sàn độ trễ của bạn. Thu nhỏ nó.
- **Không ghim threads.** callback âm thanh trên thread ưu tiên thấp hơn giao diện người dùng = trục trặc khi tải.
- **TTS đoạn quá nhỏ.** Các đoạn dưới 200 ms làm cho vocoder artifacts nghe được. Các khối 320 ms là điểm ngọt ngào.
- **Không có bộ đệm jitter.** Mạng thực sự bồn chồn; mà không làm mịn, bạn sẽ nhận được pops.
- **Xử lý lỗi một lần.** pipelines âm thanh phải chống va chạm. Một ngoại lệ giết chết session.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-realtime-designer.md`. Thiết kế pipeline âm thanh thời gian thực với ngân sách độ trễ cụ thể cho mỗi giai đoạn.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Mô phỏng bộ đệm vòng + VAD năng lượng; In độ trễ giai đoạn cho luồng 10 giây giả mạo.
2. **Trung bình.** Sử dụng `sounddevice`, xây dựng vòng lặp truyền qua processes micrô của bạn trong khung hình 20 ms và in trạng thái VAD ở mỗi khung hình.
3. **Khó.** Xây dựng một bài kiểm tra tiếng vang song công hoàn toàn với `aiortc`: trình duyệt → WebRTC → Python → trình duyệt WebRTC →. Đo độ trễ giữa kính với kính với xung 1 kHz.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bộ đệm vòng | Hàng đợi tròn | FIFO có kích thước cố định, không khóa (hoặc khóa SPSC) cho khung âm thanh. |
| VAD | Cổng im lặng | Model hoặc heuristic đánh dấu lời nói so với không nói. |
| Streaming ASR | STT thời gian thực | Phát ra một phần văn bản khi âm thanh đến; có giới hạn nhìn về phía trước. |
| Bộ đệm jitter | Mạng mượt mà hơn | Hàng đợi sắp xếp lại các gói không theo thứ tự; 60–80 ms điển hình. |
| AEC | Hủy tiếng vang | Trừ đường dẫn phản hồi từ loa đến micrô. |
| Sà lan vào | Người dùng làm gián đoạn | Hệ thống phát hiện giọng nói của người dùng giữa TTS; phải hủy phát lại. |
| Song công hoàn toàn | Đồng thời cả hai cách | Người dùng và bot có thể nói chuyện cùng một lúc; Moshi là song công hoàn toàn. |

## Đọc thêm

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) - Whisper gần như streaming đoạn.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) - độ trễ 200 ms song công.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) - agent orchestration âm thanh production.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — dưới 1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) - hủy echo theo mã nguồn mở.
