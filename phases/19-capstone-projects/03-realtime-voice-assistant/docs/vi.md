# Capstone 03 - Trợ lý giọng nói thời gian thực (ASR đến LLM đến TTS)

> Một agent thoại cảm thấy phù hợp có độ trễ từ đầu đến cuối dưới 800ms, biết khi nào bạn ngừng nói, xử lý việc xâm nhập và có thể gọi một công cụ mà không bị đình trệ. Retell, Vapi, LiveKit Agents và Pipecat đều đạt tiêu chuẩn này vào năm 2026. Chúng làm điều đó với cùng một hình dạng: streaming ASR, máy dò rẽ, streaming LLM và streaming TTS, tất cả đều được kết nối thông qua WebRTC với ngân sách độ trễ mạnh mẽ ở mỗi bước nhảy. Xây dựng một, đo WER và MOS và tỷ lệ cắt sai, và chạy nó theo gói loss.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (agent + pipeline), TypeScript (web client)
**Kiến thức tiên quyết:** Giai đoạn 6 (giọng nói và âm thanh), Giai đoạn 7 (transformers), Giai đoạn 11 (kỹ thuật LLM), Giai đoạn 13 (công cụ), Giai đoạn 14 (agents), Giai đoạn 17 (cơ sở hạ tầng)
**Các giai đoạn thực hiện: **P6 · P7 · P11 · P13 · P14 · Trang 17
**Thời lượng:** 30 giờ

## Vấn đề

Giọng nói là danh mục UX AI chuyển động nhanh nhất trong giai đoạn 2025-2026. Trần kỹ thuật giảm mỗi quý. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0 và Pipecat 0.0.70 đều đưa âm thanh đầu tiên dưới 800ms trong tầm tay. Thanh không chỉ có độ trễ. Đó là cảm giác tương tác: không cắt đứt người dùng, không bị cắt, phục hồi sau khi bị gián đoạn giữa câu, gọi công cụ giữa cuộc trò chuyện mà không làm dừng âm thanh, sống sót qua mạng di động bồn chồn.

Bạn không thể đạt được điều đó bằng cách ghép ba cuộc gọi REST. Kiến trúc được quy trình streaming từ đầu đến cuối. Xây dựng nó và các chế độ lỗi trở nên hiển thị: VAD được điều chỉnh để kích hoạt âm thanh điện thoại trên TV nền, một máy dò rẽ chờ dấu câu không bao giờ đến, một TTS đệm 400ms trước khi phát ra. Điểm mấu chốt là khắc phục từng cái này khi tải và xuất bản báo cáo độ trễ và chất lượng.

## Khái niệm

pipeline có năm giai đoạn streaming: **audio in** (WebRTC từ trình duyệt hoặc PSTN), **ASR** (streaming bản ghi một phần từ Deepgram Nova-3 hoặc thì thầm nhanh hơn), **phát hiện rẽ** (VAD cộng với một model phát hiện rẽ nhỏ đọc bản ghi một phần cho các tín hiệu hoàn thành), **LLM** (streaming tokens ngay sau khi lượt được đánh giá là hoàn tất), **TTS** (streaming âm thanh ra trong vòng ~200ms của LLM token đầu tiên).

Ba mối quan tâm xuyên suốt. **Xà nhà**: khi người dùng bắt đầu nói trong khi agent đang nói, TTS sẽ hủy và ASR bắt đầu ngay lập tức. **Sử dụng công cụ**: các cuộc gọi chức năng giữa cuộc trò chuyện (thời tiết, lịch) phải chạy trên một kênh bên mà không làm dừng âm thanh; agent điền trước token xác nhận ("một giây...") nếu độ trễ vượt quá 300ms. **Áp suất ngược**: dưới loss gói, một phần bản ghi được giữ, VAD nâng ngưỡng cổng giọng nói và agent tránh nói qua một tin nhắn không được xác nhận.

Thanh đo là định lượng. WER dưới 8% trên Hamming VAD benchmark ở 15 dB SNR. Đầu ra âm thanh p50 dưới 800ms trên 100 cuộc gọi được đo. Tỷ lệ cắt sai dưới 3%. MOS trên 4.2 trên TTS. 50 lệnh gọi đồng thời trên một g5.xlarge. Những con số này là sản phẩm được giao.

## Kiến trúc

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- Transport: LiveKit Agents 1.0 (WebRTC) cộng với Twilio PSTN gateway; Pipecat 0.0.70 làm framework thay thế
- ASR: Deepgram Nova-3 (streaming, dưới 300ms đầu tiên một phần) hoặc Whisper-v3-turbo tự lưu trữ thì thầm nhanh hơn
- VAD: Silero VAD v5 cộng với máy dò rẽ LiveKit (transformer nhỏ đọc bản ghi một phần)
- LLM: OpenAI GPT-4o-thời gian thực để tích hợp chặt chẽ, Gemini 2.5 Flash Live hoặc xếp tầng Claude Haiku 4.5 (hoàn thành streaming, đường dẫn âm thanh riêng biệt)
- TTS: Cartesia Sonic-2 (byte đầu tiên thấp nhất), ElevenLabs Flash v3 hoặc Orpheus mã nguồn mở để tự lưu trữ
- Công cụ: Kênh bên FastMCP cho weather/calendar/booking; agent phát ra trước chất độn nếu công cụ mất >300ms
- Observability: spans giọng nói OpenTelemetry, traces giọng nói Langfuse với phát lại âm thanh
- Triển khai: g5.xlarge đơn (VRAM 24GB) cho Whisper + Orpheus tự lưu trữ; APIs được lưu trữ để có độ trễ thấp nhất

## Tự xây dựng

1. **WebRTC session.** Dựng phòng LiveKit và ứng dụng web truyền phát âm thanh micrô. Trên server, hãy gắn một agent worker nối phòng.

2. **ASR streaming.** Nạp khung hình PCM 20ms vào Deepgram Nova-3 (hoặc thì thầm nhanh hơn trên GPU). Đăng ký bảng điểm một phần và cuối cùng. Ghi nhật ký độ trễ trên mỗi phần.

3. **VAD và máy dò rẽ.** Chạy Silero VAD v5 trên luồng khung. Trong sự kiện kết thúc lời nói, hãy kích hoạt trình phát hiện lượt LiveKit đối với bản chép lời một phần mới nhất. Chỉ commit "hoàn thành" khi VAD nói im lặng trong 500ms và điểm hoàn thành của máy dò rẽ > 0,6.

4. **LLM luồng.** Khi hoàn tất, hãy bắt đầu cuộc gọi LLM với cuộc trò chuyện đang chạy cộng với bản chép lời cuối cùng. Phát trực tuyến tokens ra. Ở token đầu tiên, hãy giao cho TTS.

5. **TTS luồng.** Cartesia Sonic-2 truyền lại các đoạn âm thanh. Đoạn đầu tiên phải rời khỏi server trong vòng 200ms kể từ LLM token đầu tiên. Phát các đoạn vào phòng LiveKit; ứng dụng phát thông qua bộ đệm jitter WebRTC.

6. **Sà lan.** Khi VAD phát hiện giọng nói của người dùng mới trong khi TTS đang phát, hãy hủy luồng TTS ngay lập tức, bỏ đầu ra LLM còn lại và trang bị lại ASR. Xuất bản `tts_canceled` span.

7. **Kênh bên công cụ.** Đăng ký thời tiết và lịch làm công cụ gọi chức năng. Khi được gọi, hãy kích hoạt cuộc gọi đồng thời; nếu nó không giải quyết trong vòng 300ms, hãy yêu cầu LLM phát ra "một giây, hãy để tôi kiểm tra" làm chất độn; Tiếp tục sau khi công cụ quay trở lại.

8. **Đánh giá harness.** Ghi lại 100 cuộc gọi. Tính toán WER (so với bản ghi bị giữ lại), tỷ lệ cắt sai (TTS bị hủy khi người dùng đang ở giữa câu), đầu ra âm thanh p50, TTS MOS (con người hoặc NISQA) và kiểm tra jitter-loss (giảm 3% gói).

9. **Kiểm tra tải.** Lái 50 cuộc gọi đồng thời trên một g5.xlarge duy nhất với một người gọi tổng hợp. Đo lường đầu ra âm thanh đầu tiên p95 duy trì.

## Ứng dụng

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Sản phẩm bàn giao

`outputs/skill-voice-agent.md` là sản phẩm được giao. Với một miền (hỗ trợ khách hàng, lập lịch hoặc ki-ốt), nó sẽ tạo ra một agent LiveKit với ASR/VAD/LLM/TTS pipeline được điều chỉnh theo thanh đo lường. Bảng đánh giá:

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Độ trễ từ đầu đến cuối | P50 đầu ra âm thanh dưới 800 mili giây trên 100 cuộc gọi được ghi âm |
| 20 | Chất lượng quay vòng | Tỷ lệ cắt sai dưới 3% trên Hamming VAD benchmark |
| 20 | Tính đúng đắn khi sử dụng công cụ | Các cuộc gọi công cụ giữa cuộc trò chuyện trả về dữ liệu phù hợp mà không làm gián đoạn âm thanh |
| 20 | Độ tin cậy theo gói loss | WER và ổn định quay với 3% thả gói được tiêm |
| 15 | Đánh giá tính đầy đủ harness | Các phép đo có thể tái tạo với config công cộng |
| **100** |||

## Bài tập

1. Hoán đổi Deepgram Nova-3 để lấy turbo v3 thì thầm nhanh hơn trên g5.xlarge. Đo độ trễ và khoảng cách WER. Xác định các quyết định CPU vs GPU quan trọng.

2. Thêm policy trọng tài gián đoạn: agent làm gì khi người dùng xông vào trong khi gọi công cụ? So sánh ba policies (hủy cứng, kết thúc công cụ sau đó dừng, xếp hàng lượt tiếp theo).

3. Chạy thử nghiệm phát hiện rẽ đối nghịch: cho người dùng tạm dừng dài giữa câu. Điều chỉnh ngưỡng im lặng VAD và ngưỡng điểm số của máy dò rẽ để có mức cắt sai thấp nhất mà không vượt quá 900ms.

4. Triển khai cùng một agent trên PSTN qua Twilio. So sánh PSTN đầu ra âm thanh đầu tiên với WebRTC. Giải thích sự khác biệt giữa bộ đệm jitter và codec.

5. Thêm tính năng phát hiện hoạt động giọng nói cho các ngôn ngữ không phải tiếng Anh (tiếng Nhật, tiếng Tây Ban Nha). Đo tỷ lệ trigger sai của Silero VAD v5 so với các tinh chỉnh theo ngôn ngữ cụ thể.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Phát hiện rẽ | "Kết thúc lời nói" | Bộ phân loại, với sự im lặng của VAD và một phần bản ghi, quyết định người dùng đã nói xong |
| Sà lan vào | "Xử lý gián đoạn" | Hủy TTS phát lại khi VAD phát hiện giọng nói của người dùng mới |
| Đầu ra âm thanh đầu tiên | "Độ trễ" | Thời gian từ khi người dùng ngừng nói đến gói âm thanh đầu tiên rời khỏi server |
| VAD | "Cổng phát biểu" | Model phân loại khung âm thanh là lời nói và im lặng; Silero VAD v5 là mặc định năm 2026 |
| Bộ đệm jitter | "Làm mượt âm thanh" | Bộ đệm phía máy khách giữ các gói trong thời gian ngắn để hấp thụ variance mạng |
| Chất độn | "Lời cảm ơn token" | Cụm từ ngắn mà agent phát ra để tránh im lặng khi công cụ chậm |
| MOS | "Điểm ý kiến trung bình" | Đánh giá chất lượng giọng nói tri giác; NISQA là proxy tự động |

## Đọc thêm

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — tham khảo WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) - streaming agent framework Python thay thế
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — tài liệu tham khảo cho models giọng nói tích hợp
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR tham khảo
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — model tham chiếu VAD
- [Cartesia Sonic-2](https://docs.cartesia.ai) — tham chiếu TTS độ trễ thấp
- [Retell AI architecture](https://docs.retellai.com) — kiến trúc agent giọng nói production
- [Vapi.ai production stack](https://docs.vapi.ai) — tham chiếu production thay thế
