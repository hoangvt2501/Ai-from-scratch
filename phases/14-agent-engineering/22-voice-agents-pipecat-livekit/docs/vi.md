# Voice Agents: Pipecat và LiveKit

> Voice agents là danh mục class production đầu tiên vào năm 2026. Pipecat cung cấp cho bạn pipeline dựa trên khung hình Python (VAD → STT → LLM → TTS → transport). LiveKit Agents kết nối AI models với người dùng qua WebRTC. Production độ trễ mục tiêu ở mức 450–600ms từ đầu đến cuối cho stacks cao cấp.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 14 · 12 (Mẫu quy trình làm việc)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả pipeline dựa trên khung của Pipecat: DOWNSTREAM (nguồn→sink) và UPSTREAM (điều khiển).
- Đặt tên cho giọng nói chuẩn pipeline stages và giai đoạn nào transports Pipecat hỗ trợ.
- Giải thích hai agent classes giọng nói của LiveKit Agents (MultimodalAgent, VoicePipelineAgent) và khi nào mỗi loại phù hợp.
- Tóm tắt kỳ vọng về độ trễ production năm 2026 và cách chúng thúc đẩy các lựa chọn kiến trúc.

## Vấn đề

Voice agents không phải là một vòng lặp văn bản với TTS được gắn vào. Ngân sách độ trễ rất khủng khiếp (~600ms), âm thanh một phần là mặc định, phát hiện lượt là model và phạm vi transports từ SIP điện thoại đến WebRTC. Bạn xây dựng một pipeline dựa trên khung (Pipecat) hoặc bạn dựa vào một nền tảng (LiveKit).

## Khái niệm

### Mèo ống (pipecat-ai/pipecat)

- Python pipeline framework dựa trên khung.
- `Frame` → `FrameProcessor` xích.
- Hai hướng dòng chảy:
  - **DOWNSTREAM **- nguồn → chìm (âm thanh vào, TTS ra).
  - **NGƯỢC DÒNG** - phản hồi và kiểm soát (hủy bỏ, số liệu, sà lan).
- `PipelineTask` quản lý vòng đời với các sự kiện (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) và trình quan sát cho metrics/tracing/RTVI.

pipeline tiêu biểu:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Hàng ngày, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows thêm các cuộc hội thoại có cấu trúc (máy trạng thái). Pipecat Cloud là runtime được quản lý.

### LiveKit Agents (livekit/agents)

- Cầu nối AI models với người dùng qua WebRTC.
- Các khái niệm chính: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Hai agent classes thoại:
  - **MultimodalAgent** — âm thanh trực tiếp qua OpenAI Realtime hoặc tương đương.
  - **VoicePipelineAgent** — STT → LLM → TTS tầng; cung cấp quyền kiểm soát cấp độ văn bản.
- Phát hiện ngã rẽ ngữ nghĩa thông qua transformer model.
- Tích hợp MCP gốc.
- Điện thoại qua SIP.
- 50+ models không có phím API qua LiveKit Inference; 200+ hơn nữa thông qua plugins.

### Nền tảng thương mại

Vapi (~450–600ms trên stack cao cấp được tối ưu hóa) và Retell (~600ms end-to-end trên 180 lệnh gọi thử nghiệm) được xây dựng dựa trên những điều này. Chọn một nền tảng khi bạn muốn có một stack thoại được quản lý mà không cần nhóm WebRTC.

### Mô hình này sai ở đâu

- **Không xử lý sà lan.** Người dùng làm gián đoạn; agent tiếp tục nói. Yêu cầu các khung hình hủy UPSTREAM trong Pipecat, tương đương trong LiveKit.
- **Sự tự tin của STT bị bỏ qua.** Bảng điểm có độ tin cậy thấp được cung cấp cho các LLM như thể phúc âm. Cổng về sự tự tin hoặc yêu cầu xác nhận.
- **TTS Cắt giữa câu.** Khi pipeline hủy bỏ giữa chừng, TTS cần biết hoặc cắt âm thanh.
- **Ngân sách độ trễ bị bỏ qua.** Mọi thành phần đều thêm 50–200 mili giây. Tổng chuỗi của bạn trước khi shipping.

### Độ trễ điển hình năm 2026

- VAD: 20–60 mili giây
- STT một phần: 100–250ms
- LLM token đầu tiên: 150–400ms
- TTS âm thanh đầu tiên: 100–200ms
- Transport RTT: 30–80ms

450–600ms từ đầu đến cuối là cao cấp. 800–1200ms là phổ biến. Bất cứ điều gì > 1500ms đều cảm thấy bị hỏng.

## Tự xây dựng

`code/main.py` là một pipeline đồ chơi dựa trên khung với:

- `Frame` loại (âm thanh, bản ghi, văn bản, tts_audio, điều khiển).
- `Processor` giao diện với `process(frame)`.
- Một pipeline năm giai đoạn (VAD → STT → LLM → TTS → transport) làm bộ xử lý theo kịch bản.
- Một khung hủy UPSTREAM để chứng minh sà lan.

Chạy nó:

```
python3 code/main.py
```

trace hiển thị dòng chảy bình thường và hủy sà lan dừng lại TTS giữa chừng.

## Ứng dụng

- **Pipecat** để kiểm soát hoàn toàn — bộ xử lý tùy chỉnh, nhà cung cấp ưu tiên Python, có thể cắm được.
- **LiveKit Agents** để triển khai và điện thoại ưu tiên WebRTC.
- **Vapi / Retell** để agents thoại được lưu trữ mà không cần nhóm WebRTC.
- **OpenAI Thời gian thực / Gemini Trực tiếp** để audio-in/audio-out trực tiếp (MultimodalAgent).

## Sản phẩm bàn giao

`outputs/skill-voice-pipeline.md` giàn giáo một pipeline giọng nói hình Pipecat với VAD + STT + LLM + TTS + transport cộng với khả năng xử lý sà lan.

## Bài tập

1. Thêm trình quan sát chỉ số vào pipeline đồ chơi của bạn: đếm khung hình trên mỗi giai đoạn mỗi giây. Độ trễ tích lũy ở đâu?
2. Triển khai STT được kiểm soát độ tin cậy: dưới ngưỡng, yêu cầu "bạn có thể lặp lại điều đó không?"
3. Thêm phát hiện lượt ngữ nghĩa: quy tắc đơn giản - nếu bản chép lời kết thúc bằng "?", kết thúc lượt.
4. Đọc tài liệu transport của Pipecat. Hoán đổi transport stdlib cho config SmallWebRTCTransport (sơ khai).
5. Đo lường tầng OpenAI Thời gian thực so với STT+LLM+TTS trên cùng một truy vấn. Kiểm soát cấp văn bản có chi phí độ trễ là bao nhiêu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Khung | "Sự kiện" | Đơn vị dữ liệu được nhập trong pipeline (âm thanh, bản ghi, văn bản, điều khiển) |
| Bộ xử lý | "Sân khấu Pipeline" | Bộ xử lý với process (khung) |
| HẠ LƯU | "Dòng chảy chuyển tiếp" | Nguồn để chìm: âm thanh vào, giọng nói ra |
| NGƯỢC DÒNG | "Luồng phản hồi" | Kiểm soát: hủy, số liệu, sà lan |
| VAD | "Phát hiện hoạt động giọng nói" | Phát hiện khi người dùng đang nói |
| Phát hiện rẽ ngữ nghĩa | "Kết thúc ngã rẽ thông minh" | Quyết định dựa trên Model rằng người dùng đã hoàn thành |
| Đại lý đa phương thức | "agent âm thanh trực tiếp" | Âm thanh vào, âm thanh ra; Không có văn bản ở giữa |
| VoicePipelineAgent | "Thác agent" | STT + LLM + TTS; Kiểm soát mức văn bản |

## Đọc thêm

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — pipeline dựa trên khung, bộ xử lý transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) - WebRTC + primitives thoại
- [Vapi](https://vapi.ai/) — nền tảng thoại được quản lý
- [Retell AI](https://www.retellai.com/) — giọng nói được quản lý, được đo điểm chuẩn độ trễ
