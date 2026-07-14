# Phát hiện hoạt động giọng nói và lấy lượt - Silero, Cobra và Thủ thuật xả nước

> Mọi giọng nói agent sống hay chết dựa trên hai quyết định: người dùng có đang nói bây giờ không, và họ đã hoàn thành chưa? VAD trả lời câu đầu tiên. Phát hiện rẽ (VAD + im lặng-nôn nao + endpoint model ngữ nghĩa) trả lời câu hỏi thứ hai. Nếu bạn làm sai và trợ lý của bạn sẽ cắt người dùng hoặc không bao giờ im lặng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 11 (Âm thanh thời gian thực), Giai đoạn 6 · 12 (Trợ lý giọng nói)
**Thời lượng:** ~45 phút

## Vấn đề

Ba quyết định riêng biệt mà một giọng nói agent đưa ra trên mỗi đoạn 20 mili giây:

1. **Đây có phải là bài phát biểu khung không?** - VAD. Nhị phân, mỗi khung hình.
2. **Người dùng đã bắt đầu một lời nói mới chưa?** — phát hiện khởi phát.
3. **Người dùng đã hoàn thành chưa?** — end-pointing (turn-end).

Câu trả lời ngây thơ (ngưỡng năng lượng) không thành công đối với bất kỳ nhiễu nào - giao thông, bàn phím, tiếng bập bẹ của đám đông. Câu trả lời năm 2026: Silero VAD (mở, học sâu) + model phát hiện lượt (điểm cuối ngữ nghĩa) + nôn nao im lặng được hiệu chỉnh VAD.

## Khái niệm

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### Tầng VAD ba tầng

**Cấp 1: cổng năng lượng.** Rẻ nhất. Ngưỡng RMS ở -40 dBFS. Lọc im lặng rõ ràng nhưng kích hoạt bất kỳ nhiễu nào trên ngưỡng.

**Cấp 2: Silero VAD **(2020-2026, MIT). 1 triệu parameters. Được huấn luyện trên 6000+ ngôn ngữ. Chạy trong ~1 ms trên khối 30 ms trên một CPU thread. 87,7% TPR ở 5% FPR. Mặc định mã nguồn mở.

**Bậc 3: phát hiện rẽ ngữ nghĩa.** model phát hiện lượt của LiveKit (2024-2026) hoặc bộ phân loại nhỏ của riêng bạn. Phân biệt "tạm dừng giữa câu" với "nói xong". Sử dụng ngữ cảnh ngôn ngữ (ngữ điệu + từ gần đây), không chỉ im lặng.

### Key parameters và giá trị mặc định của chúng

- **Ngưỡng.** Silero xuất ra xác suất; Phân loại giọng nói ở &gt; 0,5 (mặc định) hoặc &gt; 0,3 (nhạy cảm). Ngưỡng thấp hơn = ít clip từ đầu tiên hơn, nhiều dương tính giả hơn.
- **Thời lượng phát biểu tối thiểu.** Từ chối bài phát biểu ngắn hơn 250 mili giây - thường là tiếng ho hoặc nhiễu của ghế.
- **Im lặng nôn nao (điểm cuối).** Sau khi VAD trở về 0, đợi 500-800 ms trước khi tuyên bố kết thúc lượt. Quá ngắn → làm gián đoạn người dùng. Quá lâu → cảm thấy chậm chạp.
- **Bộ đệm trước cuộn.** Giữ âm thanh 300-500 ms trước khi VAD kích hoạt. Ngăn chặn "hey" bị cắt.

### Thủ thuật xả nước (Kyutai 2025)

Streaming STT models có độ trễ dự kiến (500 ms đối với Kyutai STT-1B, 2,5 giây đối với STT-2,6B). Thông thường, bạn sẽ đợi rất lâu sau khi kết thúc bài phát biểu để nhận bản ghi. Thủ thuật xả: khi VAD kích hoạt kết thúc giọng nói, **gửi tín hiệu xả đến STT **buộc đầu ra ngay lập tức. STT processes ở ~4× thời gian thực, vì vậy bộ đệm 500 ms kết thúc trong ~125 ms.

End-to-end: 125 ms VAD + flush STT = độ trễ đàm thoại.

### So sánh VAD năm 2026

| VAD | TPR @ 5% FPR | Độ trễ | Giấy phép |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 mili giây | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 mili giây | Tiểu bang MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 mili giây | thương mại |
| Phân đoạn Pyannote | 95% | ~10 mili giây | MIT-ish |

Silero là mặc định đúng. Cobra là sự tuân thủ / nâng cấp accuracy. VAD chỉ sử dụng năng lượng không có chỗ vào năm 2026 production.

## Tự xây dựng

### Bước 1: cổng năng lượng

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Bước 2: Silero VAD trong Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Bước 3: máy trạng thái kết thúc

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Bước 4: bộ xương thủ thuật xả

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) phải hỗ trợ flush để điều này hoạt động. Whisper streaming thì không - nó dựa trên khối và luôn chờ đợi các khối.

## Ứng dụng

| Tình huống | Lựa chọn VAD |
|-----------|-----------|
| Mở, nhanh, chung | Silero VAD |
| Tổng đài thương mại | Rắn hổ mang VAD |
| Trên thiết bị (điện thoại) | Silero VAD ONNX |
| Nghiên cứu / diarization | Phân đoạn Pyannote |
| Dự phòng không phụ thuộc | WebRTC VAD (kế thừa) |
| Cần chất lượng kết thúc lượt | Máy dò rẽ Silero + LiveKit phân lớp |

Nguyên tắc chung: không bao giờ ship VAD chỉ sử dụng năng lượng trừ khi bạn thực sự không có lựa chọn nào khác.

## Cạm bẫy

- **Ngưỡng cố định.** Hoạt động trong môi trường yên tĩnh, không thành công trong nhiễu. Hiệu chỉnh trên thiết bị hoặc chuyển sang Silero.
- **Nôn nao im lặng quá ngắn.** Agent ngắt lời giữa câu. 500-800 ms là điểm ngọt ngào cho lời nói đàm thoại.
- **Nôn nao quá lâu.** Cảm thấy uể oải. A/B thử nghiệm với người dùng mục tiêu.
- **Không có bộ đệm pre-roll.** 200-300 ms đầu tiên của âm thanh người dùng bị mất. Luôn giữ một cuộn trước.
- **Bỏ qua điểm cuối ngữ nghĩa.** "Hmm, để tôi suy nghĩ..." chứa các khoảng dừng dài. Người dùng ghét bị cắt giữa chừng. Sử dụng máy dò rẽ của LiveKit hoặc tương tự.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-vad-tuner.md`. Chọn chiến lược VAD model, ngưỡng, nôn nao, cuộn trước và phát hiện lượt cho khối lượng công việc.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó mô phỏng trình tự lời nói + im lặng + lời nói + ho và kiểm tra ba cấp VAD.
2. **Trung bình.** Cài đặt `silero-vad`, process ghi âm 5 phút, ngưỡng điều chỉnh để giảm thiểu cả clip từ đầu tiên và triggers sai. Báo cáo precision/recall.
3. **Khó.** Xây dựng một máy dò rẽ mini: Silero VAD + MLP 3 lớp trên embeddings của 10 từ cuối cùng (sử dụng câu-transformers). Huấn luyện trên một dataset rẽ cuối được dán nhãn bằng tay. Đánh bại Silero-only 10% F1.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| VAD | Máy dò giọng nói | Nhị phân trên mỗi khung hình: đây có phải là bài phát biểu không? |
| Phát hiện rẽ | Điểm cuối | VAD + im lặng-nôn nao + endpoint ngữ nghĩa. |
| Im lặng nôn nao | Chờ sau bài phát biểu | Đã đến lúc đợi trước khi tuyên bố kết thúc ngã rẽ; 500-800 mili giây. |
| Cuộn trước | Bộ đệm trước lời nói | Giữ âm thanh 300-500 ms trước khi VAD kích hoạt. |
| Thủ thuật xả | Hack Kyutai | VAD → flush-STT → độ trễ 125 ms thay vì độ trễ 500 ms. |
| endpoint ngữ nghĩa | "Họ có ý dừng lại không?" | ML phân loại xem xét các từ, không chỉ im lặng. |
| TPR @ FPR 5% | Điểm ROC | Tiêu chuẩn VAD benchmark; 87,7% cho Silero, 50% WebRTC. |

## Đọc thêm

- [Silero VAD](https://github.com/snakers4/silero-vad) — tham chiếu mở VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — lãnh đạo accuracy thương mại.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) - thủ thuật kỹ thuật dưới 200 ms.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — điểm cuối ngữ nghĩa trong production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) - đường cơ sở kế thừa.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) - phân đoạn cấp diarization.
