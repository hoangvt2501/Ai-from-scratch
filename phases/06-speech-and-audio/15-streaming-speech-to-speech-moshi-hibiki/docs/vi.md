# Streaming Speech-to-Speech — Moshi, Hibiki và Full-Duplex Dialogue

> 2024-2026 xác định lại AI giọng nói. Moshi ships một model duy nhất nghe và nói đồng thời với độ trễ 200 ms. Hibiki thực hiện dịch lời nói thành giọng nói từng đoạn. Cả hai đều từ bỏ ASR → LLM → TTS pipeline để chuyển sang kiến trúc song công thống nhất trên tokens codec Mimi. Đây là thiết kế tham chiếu mới.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 13 (Codec âm thanh thần kinh), Giai đoạn 6 · 11 (Âm thanh thời gian thực), Giai đoạn 7 · 05 (Full Transformer)
**Thời lượng:** ~75 phút

## Vấn đề

Mỗi giọng nói agent xây dựng từ Bài 11 + 12 đều có sàn độ trễ cơ bản khoảng 300-500 ms: VAD cháy, STT processes LLM lý do TTS tạo ra. Mỗi giai đoạn có độ trễ tối thiểu riêng. Bạn có thể điều chỉnh và song song, nhưng hình dạng pipeline giới hạn bạn.

Moshi (Kyutai, 2024-2026) đặt ra một câu hỏi khác: nếu không có pipeline thì sao? Điều gì sẽ xảy ra nếu một model nhận âm thanh vào và phát ra âm thanh trực tiếp, liên tục, với văn bản như một "độc thoại bên trong" trung gian thay vì một giai đoạn bắt buộc?

Câu trả lời là **chuyển giọng nói thành giọng nói song công**. Độ trễ lý thuyết 160 ms (80 ms Mimi frame + 80 ms acoustic delay). Độ trễ thực tế 200 ms trên một GPU L4 duy nhất. Đó là một nửa những gì một giọng nói tốt nhất trong class agent đạt được.

## Khái niệm

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Kiến trúc Moshi

**Đầu vào.** Hai luồng codec Mimi, cả hai đều ở tần số 12,5 Hz × 8 sách mã:

- Luồng 1: âm thanh của người dùng (Mã hóa Mimi, liên tục đến)
- Luồng 2: Âm thanh của chính Moshi (do Moshi tạo ra)

**The transformer.** A 7B-parameter Temporal Transformer processes cả luồng và luồng "độc thoại nội tâm" văn bản. Ở mỗi bước 80 ms, nó:

1. Tiêu thụ người dùng mới nhất Mimi tokens (8 cuốn sách).
2. Tiêu thụ tokens Moshi Mimi gần đây nhất (8 cuốn sách, khi được sản xuất).
3. Tạo văn bản Moshi tiếp theo token (độc thoại bên trong).
4. Tạo tokens Moshi Mimi tiếp theo (8 sách mã thông qua một Depth Transformer nhỏ).

Cả ba luồng - âm thanh người dùng, âm thanh Moshi, văn bản Moshi - chạy song song. Moshi có thể nghe thấy người dùng trong khi nói; có thể tự ngắt khi người dùng làm gián đoạn; có thể kênh ngược ("MHM") mà không làm ngắt lời chính của nó.

**Độ sâu transformer.** Trong một khung, 8 codebook không được dự đoán song song - chúng có sự phụ thuộc giữa các codebook. Một "transformer độ sâu" 2 lớp nhỏ dự đoán chúng tuần tự trong vòng 80 ms. Đây là thừa số tiêu chuẩn cho LM codec AR (cũng được sử dụng bởi VALL-E, VibeVoice).

### Tại sao văn bản độc thoại bên trong lại giúp ích

Nếu không có văn bản rõ ràng, model phải ngầm model ngôn ngữ trong luồng âm thanh của nó. Cái nhìn sâu sắc của Moshi: buộc nó phát ra tokens văn bản cùng với âm thanh. Luồng văn bản về cơ bản là bản ghi những gì Moshi đang nói. Điều này cải thiện tính mạch lạc ngữ nghĩa, giúp bạn dễ dàng hoán đổi ngôn ngữ model đầu và cung cấp cho bạn bảng điểm miễn phí.

### Hibiki: streaming dịch chuyển giọng nói thành giọng nói

Cùng một kiến trúc, được huấn luyện trên các cặp dịch. Nguồn âm thanh vào, âm thanh ngôn ngữ đích, liên tục. Hibiki-Zero (Tháng 2 năm 2026) loại bỏ nhu cầu về dữ liệu training được căn chỉnh ở cấp độ từ - sử dụng dữ liệu cấp câu + học tăng cường GRPO để tối ưu hóa độ trễ.

Bốn cặp ngôn ngữ được hỗ trợ ban đầu; có thể được điều chỉnh sang một ngôn ngữ mới với ≈1000 giờ.

### stack Kyutai rộng lớn hơn (2026)

- **Moshi** — đối thoại song công (tiếng Pháp đầu tiên, tiếng Anh được hỗ trợ tốt)
- **Hibiki / Hibiki-Zero** — dịch giọng nói đồng thời
- **Kyutai STT** — streaming ASR (500 ms hoặc 2,5 giây nhìn về phía trước)
- **Kyutai Pocket TTS** — TTS 100 triệu tham số chạy trên CPU (tháng 1 năm 2026)
- **Bật tiếng** — pipeline đầy đủ kết hợp chúng trên servers công cộng

Thông lượng trên GPU L40S: 64 sessions đồng thời ở 3× thời gian thực.

### Sesame CSM - anh họ

Sesame CSM (2025) sử dụng một ý tưởng tương tự - xương sống Llama-3 với đầu codec Mimi. Nhưng CSM là một hướng (lấy ngữ cảnh + văn bản, tạo ra giọng nói) chứ không phải song công. Đó là TTS "sự hiện diện bằng giọng nói" tốt nhất trên thị trường; không hoàn toàn giống với khả năng song công hoàn toàn của Moshi.

### Số hiệu suất năm 2026

| Model | Độ trễ | Trường hợp sử dụng | Giấy phép |
|-------|---------|----------|---------|
| Moshi | 200 mili giây (L4) | Đối thoại song công tiếng Anh / tiếng Pháp | CC-BỞI 4.0 |
| Hibiki | Tốc độ khung hình 12.5 Hz | Dịch streaming tiếng Anh Pháp ↔ | CC-BỞI 4.0 |
| Hibiki-Zero | giống nhau | 5 cặp ngôn ngữ, không có dữ liệu được căn chỉnh | CC-BỞI 4.0 |
| Mè CSM-1B | TTFA 200 mili giây | TTS có điều kiện theo ngữ cảnh | Apache-2.0 |
| GPT-4o Thời gian thực | ~300 mili giây | đóng cửa, OpenAI API | thương mại |
| Gemini 2.5 Trực tiếp | ~350 mili giây | đã đóng, Google API | thương mại |

## Tự xây dựng

### Bước 1: giao diện

Moshi hiển thị một WebSocket server lấy 80 ms các đoạn âm thanh được mã hóa Mimi và trả về 80 ms các đoạn âm thanh được mã hóa Mimi. Cả hai cách. Liên tục.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Bước 2: vòng lặp song công

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Cả hai hướng chạy đồng thời. Python hợp đồng bộ hoặc hợp đồng tương lai Rust là transport tiêu chuẩn.

### Bước 3: mục tiêu training (khái niệm)

Đối với mỗi 80 ms khung hình `t`:

- Đầu vào: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Dự đoán: `moshi_text[t]`, sau đó `moshi_mimi[t, codebook_0..7]`

Văn bản được dự đoán trước âm thanh (độc thoại bên trong); Âm thanh được dự đoán tuần tự trong transformer độ sâu.

### Bước 4: Moshi thắng ở đâu và không thắng ở đâu

Moshi thắng:

- Dưới 250 ms từ đầu đến cuối trên phần cứng giá rẻ.
- Các kênh ngược tự nhiên và gián đoạn.
- Không có mã keo pipeline.

Moshi không thắng:

- Tool calling (không được huấn luyện cho nó; bạn cần một con đường LLM riêng).
- Lý luận dài (Moshi là một cuộc đối thoại kiểu 8B model, không phải Claude/GPT-4).
- accuracy thực tế về các chủ đề thích hợp.
- Hầu hết production các trường hợp sử dụng doanh nghiệp (vẫn sử dụng pipelines vào năm 2026).

## Ứng dụng

| Tình huống | Chọn |
|-----------|------|
| Đồng hành giọng nói có độ trễ thấp nhất | Moshi |
| Cuộc gọi dịch trực tiếp | Hibiki |
| Demo / nghiên cứu giọng nói | Moshi, CSM |
| agent doanh nghiệp với các công cụ | Pipeline (Bài 12), không phải Moshi |
| Giọng nói tùy chỉnh TTS trong ngữ cảnh | CSM mè |
| Chuyển giọng nói thành giọng nói, bất kỳ ngôn ngữ nào | GPT-4o Thời gian thực hoặc Gemini 2.5 Trực tiếp (thương mại) |

## Cạm bẫy

- **Giới hạn tool calling.** Moshi là một model đối thoại, không phải agent framework. Kết hợp với pipeline cho các công cụ.
- **Điều hòa giọng nói cụ thể.** Moshi sử dụng một tính cách được huấn luyện duy nhất; Nhân bản là một training chạy riêng biệt.
- **Phạm vi ngôn ngữ.** Tiếng Pháp + Tiếng Anh là tuyệt vời; những người khác hạn chế. Hibiki-Zero hữu ích, nhưng bạn vẫn cần training dữ liệu.
- **Chi phí tài nguyên.** Một session Moshi đầy đủ chứa một vị trí GPU; không phải là một mô hình triển khai tenant chia sẻ rẻ tiền.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-duplex-pipeline.md`. Chọn kiến trúc pipeline so với full-duplex cho khối lượng công việc agent giọng nói, có lý do.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó mô phỏng kiến trúc hai luồng + độc thoại bên trong một cách tượng trưng.
2. **Trung bình.** Kéo Moshi từ HuggingFace, chạy server, kiểm tra một cuộc trò chuyện. Đo độ trễ đồng hồ treo tường từ giọng nói cuối của người dùng đến khi bắt đầu phản hồi Moshi.
3. **Khó.** Lấy bài 12 của bạn pipeline agent và so sánh độ trễ của P50 so với Moshi trên 20 câu nói thử nghiệm phù hợp. Viết lên khi một pipeline chiến thắng về mặt kiến trúc.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Song công hoàn toàn | Nghe và nói cùng một lúc | Hai luồng âm thanh hoạt động đồng thời trên cùng một model. |
| Độc thoại nội tâm | Luồng văn bản của Model | Moshi phát ra tokens văn bản cùng với đầu ra âm thanh của nó. |
| Độ sâu transformer | Dự đoán giữa các sách mã | transformer nhỏ dự đoán 8 sách mã trong một khung hình 80 ms. |
| Mimi | Codec của Kyutai | 12,5 Hz × 8 sách mã; ngữ nghĩa + âm thanh; sức mạnh của Moshi. |
| Streaming S2S | Âm thanh → âm thanh trực tiếp | Từng đoạn translation/dialogue, không pipeline giai đoạn. |
| Kênh ngược | Phản ứng "Mhm" | Moshi có thể phát ra những lời cảm ơn nhỏ mà không gặp lỗi lượt. |

## Đọc thêm

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) - tờ giấy.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — streaming dịch mà không có dữ liệu được căn chỉnh.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) - Thông số kỹ thuật CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) - cài đặt + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — ngang hàng thương mại đóng.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) - STT/TTS framework dưới mui xe.
