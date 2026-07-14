# Models âm thanh-ngôn ngữ — Qwen2.5-Omni, Audio Flamingo GPT-4o Audio

> Ngôn ngữ âm thanh năm 2026 models lý do hơn lời nói + âm thanh môi trường + âm nhạc. Qwen2.5-Omni-7B khớp với Âm thanh GPT-4o trên MMAU-Pro. Audio Flamingo Next đánh bại Gemini 2.5 Pro trên LongAudioBench. Khoảng cách giữa mở và đóng về cơ bản được thu hẹp - ngoại trừ các tác vụ đa âm thanh, nơi mọi người gần như ngẫu nhiên.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 04 (ASR), Giai đoạn 12 · 03 (Tầm nhìn-Ngôn ngữ Models), Giai đoạn 7 · 10 (Transformers âm thanh)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn có 5 giây âm thanh: tiếng chó sủa, ai đó hét lên "dừng lại!", sau đó im lặng. Các câu hỏi hữu ích span nhiều trục:

- **Phiên âm.** "What was said?" — ASR lãnh thổ.
- **Lý luận ngữ nghĩa.** "Người đó có gặp nguy hiểm không?" - đòi hỏi sự hiểu biết chung về tiếng sủa + tiếng la hét + im lặng.
- **Lý luận âm nhạc.** "Nhạc cụ nào chơi giai điệu?"
- **Truy xuất âm thanh dài.** "Người hướng dẫn đã giải thích gradient descent ở đâu trong bài giảng dài 90 phút này?"

Một model duy nhất trả lời tất cả những điều này bằng một prompt là **ngôn ngữ âm thanh model **(LALM / ALM). Tách biệt với ASR thuần túy: LALM tạo ra các câu trả lời bằng ngôn ngữ tự nhiên dạng tự do, không chỉ bảng điểm.

## Khái niệm

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### Mẫu ba thành phần

Mỗi LALM 2026 đều có cùng một bộ xương:

1. **encoder âm thanh.** Whisper encoder · NHỊP · Vỗ tay · WavLM · hoặc một encoder tùy chỉnh cho mỗi model.
2. **Máy chiếu.** Tuyến tính hoặc MLP kết nối encoder features âm thanh vào không gian token embedding của LLM.
3. **LLM.** Llama / Qwen / decoder dựa trên Gemma. Lấy văn bản xen kẽ + tokens âm thanh; tạo văn bản.

Training:

- **Giai đoạn 1.** Đóng băng encoder + LLM; chỉ huấn luyện máy chiếu trên dữ liệu ASR / phụ đề.
- **Giai đoạn 2.** Đầy đủ / LoRA fine-tune về các tác vụ âm thanh theo hướng dẫn (QA, lý luận, hiểu âm nhạc).
- **Stage 3 (tùy chọn).** Voice-in / voice-out thêm một decoder giọng nói. Qwen2.5-Omni và AF3-Chat làm điều này.

### Bản đồ model 2026

| Model | Xương sống | encoder âm thanh | Phương thức đầu ra | Cách đi |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Câu 2.5-7B | Tùy chỉnh + Thì thầm | Văn bản + Giọng nói | Apache-2.0 |
| Qwen3-Omni | Câu 3 | Tập quán | Văn bản + Giọng nói | Apache-2.0 |
| Âm thanh Flamingo 3 | Câu 2 | VỖ TAY LẤY NÉT TỰ ĐỘNG | Văn bản | NVIDIA phi thương mại |
| Âm thanh Flamingo Next | Câu 2 | AF-CLAP v2 | Văn bản | NVIDIA phi thương mại |
| CÁ HỒI | Vicuna | Thì thầm + BEATs | Văn bản | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | Văn bản | Apache-2.0 |
| GAMA | Llama | AST + Q-Cựu | Văn bản | Apache-2.0 |
| Gemini 2.5 Flash/Pro (đóng) | Gemini | Độc quyền | Văn bản + Giọng nói | API |
| GPT-4o Âm thanh (đóng) | GPT-4o | Độc quyền | Văn bản + Giọng nói | API |

### Kiểm tra thực tế Benchmark (2026)

**MMAU-Pro.** 1800 cặp QA bao gồm giọng nói / âm thanh / âm nhạc / hỗn hợp. Bao gồm nhiều tập hợp con âm thanh.

| Model | Tổng thể | Bài phát biểu | Âm thanh | Âm nhạc | Đa âm thanh |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Âm thanh | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Âm thanh Flamingo 3 | ~54% | — | — | — | — |
| Âm thanh Flamingo Next | SOTA trên LongAudioBench | — | — | — | — |

**Cột đa âm thanh đang nguyền rủa tất cả mọi người.** Cơ hội ngẫu nhiên trên trắc nghiệm 4 tùy chọn = 25%; hầu hết models ghi điểm xung quanh đó. LALM vẫn phải vật lộn để so sánh hai clip.

### LALM hữu ích ở đâu vào năm 2026

- **Kiểm tra tuân thủ các bản ghi âm của trung tâm cuộc gọi.** "agent có đề cập đến việc tiết lộ bắt buộc không?"
- **Trợ năng.** Mô tả các sự kiện âm thanh cho người dùng khiếm thính (không chỉ phiên âm).
- **Kiểm duyệt nội dung.** Phát hiện ngôn ngữ bạo lực + giọng điệu đe dọa + ngữ cảnh nền.
- **Podcast / chương cuộc họp.** Tóm tắt ngữ nghĩa, không chỉ là người nói.
- **Phân tích danh mục nhạc.** "Tìm tất cả các bản nhạc có thay đổi phím phần B."

### Nơi chúng KHÔNG (chưa) hữu ích

- Lý thuyết âm nhạc chi tiết (dưới cấp hợp âm).
- Lý luận do người nói quy kết trong các cuộc trò chuyện dài (giảm sau 10 phút).
- So sánh nhiều âm thanh (22-26% hầu như không cao hơn ngẫu nhiên).
- Lý luận streaming thời gian thực (hầu hết là batch inference ngoại tuyến).

## Tự xây dựng

### Bước 1: truy vấn Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Bước 2: mẫu máy chiếu

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Đó là nó. Máy chiếu thường có 1-3 lớp tuyến tính. Training nó trên các cặp ASR (bản ghi âm →) là nhiệm vụ lý do Giai đoạn 1.

### Bước 3: đo điểm chuẩn MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Báo cáo riêng cho từng danh mục (giọng nói / âm thanh / nhạc / đa âm thanh). Các số tổng hợp ẩn nơi model không thành công.

## Ứng dụng

| Nhiệm vụ | Lựa chọn năm 2026 |
|------|-----------|
| QA âm thanh dạng tự do (mở) | Qwen2.5-Omni-7B |
| Mở tốt nhất trên âm thanh dài | Âm thanh Flamingo Next |
| Đóng tốt nhất | Gemini 2.5 Pro |
| Giọng nói vào / giọng nói ra agent | Âm thanh Qwen2.5-Omni hoặc GPT-4o |
| Lý luận âm nhạc | Âm thanh Flamingo 3 hoặc 2 (AF-CLAP chuyên dụng về âm nhạc) |
| Kiểm tra tổng đài cuộc gọi | Gemini 2.5 Pro qua API, với RAG qua tài liệu policy của bạn |

## Cạm bẫy

- **Quá tin tưởng vào nhiều âm thanh.** Nếu nhiệm vụ của bạn cần "clip nào có X", hiệu suất ngẫu nhiên là có thật.
- **Giảm chất lượng âm thanh dài.** Sau 10 phút, hầu hết các phân bổ loa của models đều bị gián đoạn. Ghi nhật ký trước (Bài 6), sau đó tóm tắt.
- **Ảo giác về sự im lặng.** Cùng một vấn đề kiểu Whisper được thừa hưởng bởi các LALM sử dụng Whisper encoder. Cổng VAD.
- **Benchmark chọn anh đào.** Các bài đăng trên blog của nhà cung cấp làm nổi bật các danh mục trường hợp tốt nhất. Tự chạy tập hợp con đa âm thanh MMAU-Pro.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-alm-picker.md`. Chọn LALM + benchmark tập con + phương thức đầu ra (văn bản so với giọng nói) cho một nhiệm vụ hiểu âm thanh nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` để xem mẫu máy chiếu đồ chơi + định tuyến LALM giả mạo của (âm thanh-embedding, văn bản-tokens) → tokens đầu ra.
2. **Trung bình.** Ghi điểm Qwen2.5-Omni-7B trên 100 mục giọng nói MMAU-Pro. So sánh với con số báo cáo của bài báo.
3. **Khó khăn.** Xây dựng đường cơ sở phụ đề âm thanh tối thiểu: BEATs encoder + máy chiếu 2 lớp + Llama-3.2-1B đông lạnh. Chỉ Fine-tune máy chiếu trên AudioCaps. So sánh với SALMONN trên Clotho-AQA.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| LALM | ChatGPT âm thanh | encoder âm thanh + máy chiếu + LLM decoder. |
| Máy chiếu | Bộ chuyển đổi | features âm thanh ánh xạ MLP nhỏ vào không gian LLM embedding. |
| MMAU | Các benchmark | Các cặp âm thanh-QA 10k trên giọng nói, âm thanh, âm nhạc. |
| MMAU-Chuyên nghiệp | MMAU khó hơn | 1800 câu hỏi nặng về đa âm thanh / lý luận. |
| Băng ghế dài | Đánh giá dạng dài | Clip dài nhiều phút với các truy vấn ngữ nghĩa. |
| Giọng nói vào/giọng nói ra | Giọng nói bản địa | Model nhập giọng nói và phát ra giọng nói mà không có đường vòng văn bản. |

## Đọc thêm

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — kiến trúc tham khảo.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) - người dẫn đầu âm thanh dài mở.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) - LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) - người tiên phong encoder kép.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) - Bảng xếp hạng trực tiếp năm 2026.
