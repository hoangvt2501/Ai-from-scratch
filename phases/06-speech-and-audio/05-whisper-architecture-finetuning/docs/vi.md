# Whisper - Kiến trúc & Fine-Tuning

> Whisper là một decoder transformer encoder cửa sổ 30 giây, được huấn luyện trên 680 nghìn giờ các cặp âm thanh-văn bản được giám sát yếu đa ngôn ngữ. Một kiến trúc, nhiều tác vụ, mạnh mẽ trên 99 ngôn ngữ. Tham chiếu năm 2026 ASR.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 04 (ASR), Giai đoạn 5 · 10 (Attention), Giai đoạn 7 · 05 (Đầy đủ Transformer)
**Thời lượng:** ~75 phút

## Vấn đề

Whisper, được OpenAI phát hành vào tháng 9 năm 2022, là ASR model đầu tiên ship như một mặt hàng: dán âm thanh, nhận văn bản, 99 ngôn ngữ, mạnh mẽ với nhiễu, chạy trên máy tính xách tay. Đến năm 2024, OpenAI có shipped biến thể Large-v3 và Turbo; đến năm 2026, Whisper là đường cơ sở mặc định cho mọi thứ, từ phiên âm podcast đến trợ lý giọng nói đến phụ đề YouTube.

Nhưng Whisper không phải là một pipeline bạn có thể coi như một hộp đen mãi mãi. Sự dịch chuyển miền giết chết nó - biệt ngữ kỹ thuật, trọng âm của người nói, danh từ riêng, clip ngắn, im lặng. Bạn cần biết:

1. Nó thực sự là gì bên trong.
2. Làm thế nào để cung cấp cho nó âm thanh dạng khối, streaming hoặc dài một cách chính xác.
3. Khi nào nên fine-tune và như thế nào.

## Khái niệm

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Kiến trúc.** Tiêu chuẩn transformer encoder-decoder.

- Đầu vào: Quang phổ log-mel 30 giây, 80 mels, 10 ms hop → 3000 khung hình. Các clip ngắn hơn sẽ không có đệm, các clip dài hơn được chia nhỏ.
- Encoder: conv-downsample (sải bước 2) + khối `N` transformer. Đối với Large-v3: 32 lớp, 1280-mờ, 20 đầu.
- Decoder: `N` transformer các khối có tự attn nhân quả + attn chéo đến đầu ra encoder. Cùng kích thước với encoder.
- Đầu ra: BPE tokens trên một từ vựng 51.865 token.

Large-v3 có 1,55 tỷ tham số. Turbo sử dụng decoder 4 lớp (từ 32), độ trễ cắt giảm 8× với mức WER <1%.

**Định dạng prompt.** Whisper là một model đa nhiệm được điều khiển bởi các tokens đặc biệt trong decoder prompt:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — thẻ ngôn ngữ; buộc hành vi dịch và phiên âm.
- `<|transcribe|>` hoặc `<|translate|>` - dịch đầu ra tiếng Anh từ đầu vào bất kỳ ngôn ngữ nào hoặc nguyên văn.
- `<|notimestamps|>` — bỏ qua dấu thời gian cấp từ (nhanh hơn).

Sự prompt là thứ cho phép một người model làm nhiều nhiệm vụ. Thay đổi `<|en|>` thành `<|fr|>` và nó phiên âm tiếng Pháp.

**Cửa sổ 30 giây.** Mọi thứ được ghim vào 30 giây. Các clip dài hơn cần được chia nhỏ; các clip ngắn hơn được đệm. Windows không được phát trực tuyến nguyên bản - đây là lý do tại sao WhisperX, Whisper-Streaming và Faster-whisper tồn tại.

**Chuẩn hóa Log-mel.** `(log_mel - mean) / std` nơi các số liệu thống kê đến từ kho dữ liệu training của chính Whisper. Bạn *phải* sử dụng tiền xử lý của Whisper (`whisper.audio.log_mel_spectrogram`), không phải `librosa.feature.melspectrogram`.

### Các biến thể vào năm 2026

| Biến thể | Tham số | Độ trễ (A100) | WER (LibriSpeech-sạch) |
|---------|--------|----------------|------------------------|
| Tí hon | 39 triệu | 1× Thời gian thực | 5.4% |
| Căn cứ | 74 triệu | 1× | 4.1% |
| Nhỏ | 244 triệu | 1× | 3.0% |
| Trung bình | 769 triệu | 1× | 2.7% |
| Lớn-v3 | 1,55 tỷ | 2× | 1.8% |
| V3-turbo lớn | 809 triệu | 8× | 1.58% |
| Thì thầm-Streaming (2024) | 1,55 tỷ | streaming | 2.0% |

### Fine-tuning

Quy trình làm việc chuẩn vào năm 2026:

1. Thu thập 10–100 giờ âm thanh miền mục tiêu với bản chép lời được căn chỉnh.
2. Chạy `transformers.Seq2SeqTrainer` với `generate_with_loss` callback.
3. Hiệu quả Parameter: LoRA trên các lớp `q_proj`, `k_proj` `v_proj` attention giúp giảm bộ nhớ GPU 4× với chi phí WER <0,3.
4. Đông lạnh encoder nếu bạn có <10 giờ. Chỉ điều chỉnh decoder.
5. Sử dụng định dạng tokenizer và prompt riêng của Whisper; không bao giờ hoán đổi tokenizers.

Kết quả cộng đồng: fine-tuning Medium trong 20 giờ đọc chính tả y tế giảm WER từ 12% xuống 4,5% về từ vựng y tế. Fine-tuning Turbo trong 4 giờ tiếng Iceland giảm WER từ 18% xuống 6%.

## Tự xây dựng

### Bước 1: chạy Whisper ra khỏi hộp

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Các giá trị mặc định chính mà bạn phải luôn ghi đè: `temperature=0.0` (sampling mặc định là 0.0 → 0.2 → 0.4 ... Chuỗi dự phòng), `condition_on_previous_text=False` (ngăn chặn vấn đề ảo giác xếp tầng) và `no_speech_threshold=0.6` (phát hiện im lặng).

### Bước 2: dạng dài được cắt nhỏ

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX bổ sung (1) Silero VAD gating, (2) alignment cấp từ qua wav2vec 2.0, (3) diarization qua `pyannote.audio`. Con ngựa làm việc năm 2026 cho phiên âm production.

### Bước 3: fine-tune với LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Sau đó là vòng lặp Trainer tiêu chuẩn. Checkpoint sau mỗi 1000 bước. Đánh giá với WER khi bị giữ lại.

### Bước 4: kiểm tra những gì mỗi lớp học được

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Hình dung bằng bản đồ nhiệt - bạn sẽ thấy alignment đường chéo khi decoder bước quét qua encoder khung hình. Đường chéo đó là khái niệm của Whisper về dấu thời gian từ.

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh tổng quát, ngoại tuyến | Lớn-v3-turbo qua `whisperx` |
| Di động / cạnh | Whisper-Tiny lượng tử hóa (int8) hoặc Moonshine |
| Dạng dài đa ngôn ngữ | Large-v3 thông qua `whisperx` + diarization |
| Ngôn ngữ tài nguyên thấp | Fine-tune Trung bình hoặc Turbo với LoRA |
| Streaming (độ trễ 2 giây) | Whisper-Streaming hoặc vẹt đuôi dài-TDT |
| Dấu thời gian cấp từ | WhisperX (bắt buộc alignment qua wav2vec 2.0) |

`faster-whisper` (phần phụ trợ CTranslate2) là CPU+GPU inference runtime nhanh nhất vào năm 2026 - nhanh hơn 4× so với vani với đầu ra giống hệt nhau.

## Những cạm bẫy vẫn ship vào năm 2026

- **Văn bản ảo giác về im lặng.** Whisper được huấn luyện về phụ đề bao gồm "Cảm ơn vì đã xem!", "Đăng ký!", lời bài hát. Luôn luôn cổng VAD trước khi gọi.
- **`condition_on_previous_text` thác nước.** Một ảo giác làm ô nhiễm windows tiếp theo. Đặt `False` trừ khi bạn cần sự trôi chảy trên các khối.
- **Đệm clip ngắn.** Một clip dài 2 giây được đệm đến 30 giây có thể gây ảo giác trong sự im lặng theo dõi. Sử dụng `pad=False` hoặc cổng VAD.
- **Số liệu thống kê mel sai.** Sử dụng mels của librosa thay vì của Whisper tạo ra đầu ra gần như ngẫu nhiên. Sử dụng `whisper.audio.log_mel_spectrogram`.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-whisper-tuner.md`. Thiết kế fine-tune hoặc inference pipeline Whisper cho một miền nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó mã hóa một prompt kiểu Whisper, tính toán ngân sách hình dạng được giải mã và in lịch trình đoạn cho một clip dài 10 phút.
2. **Trung bình.** Cài đặt `faster-whisper`, phiên âm podcast dài 10 phút, so sánh WER với bản ghi của con người. Hãy thử `language="auto"` so với `language="en"` cưỡng bức.
3. **Khó.** Sử dụng HF `datasets`, chọn một ngôn ngữ mà Whisper gặp khó khăn (ví dụ: tiếng Urdu), fine-tune Trung bình với LoRA trong 2 epochs trên 2 giờ và báo cáo WER delta.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Cửa sổ 30 giây | Giới hạn của Whisper | Nắp đầu vào cứng; âm thanh dài hơn. |
| SOT | Bắt đầu bảng điểm | `<\ | startofbản ghi\ | >' bắt đầu decoder prompt. |
| Dấu thời gian token | alignment thời gian | Mỗi độ lệch 0,02 giây là một token đặc biệt trong từ vựng 51k. |
| Tăng áp | Biến thể nhanh | 4-decoder lớp, nhanh hơn 8×, hồi quy WER <1%. |
| Thì thầmX | Giấy bọc dạng dài | VAD + Whisper + wav2vec alignment + diarization. |
| LoRA fine-tune | Điều chỉnh hiệu quả | Thêm bộ điều hợp cấp thấp vào attention; huấn luyện ~0,3% thông số. |
| Ảo giác | Thất bại thầm lặng | Whisper tạo ra tiếng Anh lưu loát từ noise/silence. |

## Đọc thêm

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) - kiến trúc ban đầu và công thức training.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) - decoder 4 lớp, tăng tốc 8×.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — dạng dài, căn chỉnh từ, diarized.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 hỗ trợ, nhanh hơn 4×.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — hướng dẫn LoRA chuẩn / FT đầy đủ.
