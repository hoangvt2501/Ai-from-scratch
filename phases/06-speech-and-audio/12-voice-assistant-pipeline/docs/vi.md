# Xây dựng trợ lý giọng nói Pipeline - Giai đoạn 6 Capstone

> Tất cả mọi thứ từ bài 01-11, được khâu lại với nhau. Xây dựng một trợ lý giọng nói lắng nghe, suy luận và nói lại. Vào năm 2026, đó là một vấn đề kỹ thuật đã được giải quyết, không phải là một vấn đề nghiên cứu - nhưng các chi tiết tích hợp quyết định liệu nó có ships hay không.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 04, 05, 06, 07, 11; Giai đoạn 11 · 09 (Function Calling); Giai đoạn 14 · 01 (Agent Vòng lặp)
**Thời lượng:** ~120 phút

## Vấn đề

Xây dựng trợ lý đầu cuối:

1. Chụp đầu vào micrô (đơn âm 16 kHz).
2. Phát hiện start/end giọng nói của người dùng.
3. Phiên âm streaming.
4. Chuyển bản chép lời đến một LLM có thể gọi các công cụ (hẹn giờ, thời tiết, lịch).
5. Luồng LLM văn bản đến một TTS.
6. Phát lại âm thanh cho người dùng.
7. Dừng nếu người dùng làm gián đoạn giữa phản hồi.

Mục tiêu độ trễ: đầu tiên TTS byte âm thanh trong vòng 800 ms kể từ khi người dùng kết thúc lời nói của họ trên máy tính xách tay CPU. Mục tiêu chất lượng: không bỏ lỡ từ, không có phụ đề ảo giác khi im lặng, không rò rỉ nhân bản giọng nói, không có prompt tiêm thành công.

## Khái niệm

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Bảy thành phần

1. **Thu âm thanh.** Mic → đơn âm 16 kHz → các đoạn 20 ms. Thường `sounddevice` ở Python hoặc AudioUnit/ALSA/WASAPI bản địa ở production.
2. **VAD (Bài 11).** Silero VAD @ ngưỡng 0.5, bài phát biểu tối thiểu 250 ms, im lặng treo trên 500 ms. Tín hiệu "bắt đầu" và "kết thúc".
3. **Streaming STT (Bài 4-5).** Whisper-streaming, Parakeet-TDT hoặc Deepgram Nova-3 (API). Bảng điểm một phần + cuối cùng.
4. **LLM với tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON schema cho các công cụ. Phát trực tiếp tokens.
5. **Streaming TTS (Bài 7).** Kokoro-82M (mở nhanh nhất) hoặc Cartesia Sonic (thương mại). Bắt đầu TTS sau 20 LLM tokens.
6. **Phát lại.** Loa ra; opus-encode cho các mạng băng thông thấp.
7. **Trình xử lý gián đoạn.** Nếu VAD kích hoạt trong khi phát lại TTS, hãy dừng phát lại, hủy LLM, khởi động lại STT.

### Ba chế độ thất bại bạn sẽ gặp phải

1. **Clip từ đầu tiên.** VAD bắt đầu một nhịp quá muộn. Thiếu "hey" của người dùng. Ngưỡng bắt đầu ở 0,3, không phải 0,5.
2. **Sự nhầm lẫn làm gián đoạn phản hồi giữa chừng.** LLM tiếp tục tạo ra sau khi người dùng bị gián đoạn; Trợ lý nói chuyện qua người dùng. Nối dây VAD → hủy LLM.
3. **Ảo giác im lặng.** Tiếng thì thầm phát ra "Cảm ơn vì đã xem" trên khung hình khởi động im lặng. Luôn luôn là cổng VAD.

### stacks tham khảo production năm 2026

| Stack | Độ trễ | Giấy phép | Ghi chú |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 mili giây | API thương mại | Mặc định ngành 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 mili giây | chủ yếu mở | Thân thiện với DIY |
| Moshi (song công) | 200-300 mili giây | CC-BỞI 4.0 | Một model; Kiến trúc khác nhau, Bài 15 |
| Vapi / Retell (được quản lý) | 300-500 mili giây | thương mại | Khởi chạy nhanh nhất; Tùy chỉnh hạn chế |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | ngoại tuyến | Mở | Quyền riêng tư / cạnh |

## Tự xây dựng

### Bước 1: chụp mic bằng chunking (mã giả)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Bước 2: Chụp lượt có cổng VAD

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Bước 3: streaming → LLM → TTS STT

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Bước 4: tool calling bên trong vòng lặp LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Bước 5: Xử lý gián đoạn

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Ứng dụng

Xem `code/main.py` để biết mô phỏng có thể chạy được nối tất cả bảy thành phần bằng models sơ khai, vì vậy bạn có thể thấy hình dạng pipeline ngay cả khi không có phần cứng. Để triển khai thực tế, hãy hoán đổi sơ khai với:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` hoặc `openai-whisper`
- `openai` (`gpt-4o`) hoặc `anthropic`
- `kokoro` hoặc `cartesia`
- `sounddevice` cho I/O

## Cạm bẫy

- **Ghi nhật ký PII mãi mãi.** Âm thanh toàn lượt là PII ở hầu hết các khu vực pháp lý. Lưu giữ 30 ngày, được mã hóa ở rest.
- **Không được sà lan.** Người dùng sẽ làm gián đoạn. Trợ lý của bạn phải ngừng nói.
- **TTS chặn.** TTS đồng bộ chặn vòng lặp sự kiện. Sử dụng không đồng bộ hoặc một thread riêng biệt.
- **Không xử lý lỗi lệnh gọi công cụ.** Công cụ bị lỗi. LLM phải lấy lại lỗi + thử lại một lần, sau đó xuống cấp một cách duyên dáng.
- **Bộ lọc ảo giác quá mức.** Bộ lọc quá mức và trợ lý lặp lại "Tôi không thể giúp được điều đó." Under-filter và nó nói bất cứ điều gì. Hiệu chỉnh trên một bộ bị giữ.
- **Không có tùy chọn đánh thức.** Luôn lắng nghe là trách nhiệm bảo mật. Thêm cổng đánh thức (Porcupine hoặc openWakeWord).

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-voice-assistant-architect.md`. Với ngân sách + quy mô + ngôn ngữ + ràng buộc tuân thủ, hãy tạo ra một thông số kỹ thuật stack đầy đủ.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó mô phỏng một lượt đầy đủ từ đầu đến cuối với các mô-đun sơ khai và in độ trễ trên mỗi giai đoạn.
2. **Trung bình.** Thay thế sơ khai STT bằng một model Whisper thực trên một `.wav` được ghi sẵn. Đo WER và độ trễ từ đầu đến cuối.
3. **Khó.** Thêm tool calling: triển khai `get_weather` (bất kỳ API nào) và `set_timer`. Định tuyến LLM qua các công cụ và xác minh rằng khi người dùng nói "đặt hẹn giờ 5 phút", chức năng phù hợp sẽ kích hoạt và câu trả lời bằng giọng nói sẽ xác nhận điều đó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Xoay | Một người dùng + trợ lý khứ hồi | Một giọng nói của người dùng bị giới hạn bởi VAD + một phản hồi LLM TTS. |
| Sà lan vào | Gián đoạn | Người dùng nói trong khi trợ lý nói chuyện; Trợ lý dừng lại. |
| Đánh thức từ | "Này trợ lý" | Trình phát hiện từ khóa ngắn; Nhím, Cậu bé tuyết, openWakeWord. |
| Điểm cuối | Lượt kết thúc | VAD + quyết định im lặng tối thiểu mà người dùng đã hoàn thành. |
| Cuộn trước | Bộ đệm trước lời nói | Giữ âm thanh 200-400 ms trước khi VAD kích hoạt để tránh clip từ đầu tiên. |
| Lệnh gọi công cụ | Gọi hàm | LLM phát ra JSON; runtime công văn; kết quả cung cấp lại trong vòng lặp. |

## Đọc thêm

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — tham chiếu cấp production.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) - framework thân thiện với DIY.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — đường dẫn gốc giọng nói được quản lý.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — tài liệu tham khảo song công (Bài 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) - đánh thức từ gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
