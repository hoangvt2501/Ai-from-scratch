# Nhận dạng giọng nói (ASR) — CTC, RNN-T, Attention

> Nhận dạng giọng nói là phân loại âm thanh ở mọi bước thời gian, được dán lại với nhau bằng một chuỗi model biết tiếng Anh và im lặng. CTC, RNN-T và attention là ba cách để làm điều đó. Chọn một và hiểu tại sao.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ & Mel), Giai đoạn 5 · 08 (CNN & RNN cho văn bản), Giai đoạn 5 · 10 (Attention)
**Thời lượng:** ~45 phút

## Vấn đề

Bạn có một clip 10 kHz dài 16 giây. Bạn muốn có một chuỗi: "bật đèn nhà bếp". Thách thức là cấu trúc: khung âm thanh không căn chỉnh một-một với các ký tự. Từ "ổn" có thể mất 200 mili giây hoặc 1200 mili giây. Sự im lặng ngắt quãng lời nói. Một số âm vị dài hơn những âm vị khác. Số lượng tokens đầu ra không được biết trước.

Ba công thức giải quyết vấn đề này:

1. **CTC (Phân loại thời gian kết nối).** Phát ra xác suất token trên mỗi khung hình bao gồm một *trống*. Thu gọn các lần lặp lại và khoảng trống tại thời điểm giải mã. Không tự thoái lui, nhanh. Được sử dụng bởi wav2vec 2.0, MMS.
2. **RNN-T (Đầu dò mạng nơ-ron tuần hoàn).** Mạng chung dự đoán token tiếp theo cho khung encoder và tokens trước đó. Có thể phát trực tuyến. Được sử dụng bởi ASR trên thiết bị của Google, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder nén âm thanh thành các trạng thái ẩn, decoder tham dự chéo để tạo tokens tự hồi quy. Được sử dụng bởi Whisper, SeamlessM4T.

Vào năm 2026, SOTA WER trên LibriSpeech kiểm tra sạch là 1.4% (Parakeet-TDT-1.1B, NVIDIA) và 1.58% (Whisper-Large-v3-turbo). Sự khác biệt là rất nhỏ; Sự khác biệt về triển khai là rất lớn.

## Khái niệm

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Trực giác CTC.** Hãy để encoder xuất ra `T` phân phối cấp khung hình trên `V+1` tokens (ký tự V + trống). Đối với một chuỗi mục tiêu `y` độ dài `U < T`, bất kỳ alignment khung nào thu gọn xuống `y` đều được tính. CTC loss tổng tất cả các liên kết như vậy. Inference: argmax trên mỗi khung hình, thu gọn lặp lại, xóa khoảng trống.

Ưu điểm: không tự hồi quy, có thể phát trực tuyến, không nhìn về phía trước. Hạn chế: *giả định độc lập có điều kiện* — mỗi dự đoán khung độc lập với các dự đoán khác, vì vậy không có model ngôn ngữ nội bộ. Sửa chữa bằng LM bên ngoài thông qua beam search hoặc nhiệt hạch nông.

**Trực giác RNN-T.** Thêm một mạng *dự đoán* nhúng lịch sử token và một *trình nối* kết hợp trạng thái dự đoán với khung encoder vào một phân phối chung trên `V+1` (`+1` là rỗng / không phát ra). Rõ ràng models sự phụ thuộc có điều kiện CTC đã bỏ qua. Có thể phát trực tuyến vì mỗi bước chỉ điều kiện trên các khung hình trước đây và tokens trước đây.

Ưu điểm: có thể phát trực tuyến + LM nội bộ. Hạn chế: training phức tạp hơn và ngốn bộ nhớ hơn (mạng tinh thể 3D loss); Hạt nhân loss RNN-T là một danh mục thư viện toàn bộ.

**Attention encoder-decoder.** Encoder (6-32 transformer lớp) trên khung log-mel. Decoder (6-32 lớp transformer) tham gia chéo các đầu ra encoder để tạo tokens tự hồi quy. Không có ràng buộc alignment - attention có thể nhìn bất cứ đâu trong âm thanh. Không thể phát trực tuyến trừ khi bạn hạn chế attention (đoạn Whisper-Streaming, 2024).

Ưu điểm: chất lượng cao nhất trên ASR ngoại tuyến, dễ huấn luyện với công cụ seq2seq tiêu chuẩn. Hạn chế: độ trễ tự hồi quy tỷ lệ thuận với độ dài đầu ra; không thể phát trực tuyến nếu không có kỹ thuật.

### WER: một số

**Tỷ lệ lỗi từ** = `(S + D + I) / N`, trong đó S = thay thế, D = xóa, I = chèn, N = số từ tham chiếu. Khớp với khoảng cách chỉnh sửa Levenshtein ở cấp độ từ. Thấp hơn là tốt hơn. WER trên 20% thường không sử dụng được; dưới 5% là sự ngang bằng của con người đối với lời nói đọc. Con số năm 2026 trên benchmarks tiêu chuẩn:

| Model | Làm sạch kiểm tra LibriSpeech | Kiểm tra LibriSpeech-khác | Kích thước |
|-------|------------------------|------------------------|------|
| Vẹt đuôi dài-TDT-1.1B | 1.40% | 2.78% | Tham số 1,1 tỷ |
| Thì thầm-Lớn-v3-turbo | 1.58% | 3.03% | 809 triệu |
| Canary-1B Flash | 1.48% | 2.87% | 1 tỷ |
| M4T v2 liền mạch | 1.7% | 3.5% | 2,3 tỷ |

Tất cả những thứ này đều dựa trên encoder-decoder hoặc RNN-T. Hệ thống CTC tinh khiết (wav2vec 2.0) nằm khoảng 1,8–2,1% khi làm sạch thử nghiệm.

## Tự xây dựng

### Bước 1: giải mã CTC tham lam

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Hai quy tắc: thu gọn các lần lặp lại liên tiếp, bỏ chỗ trống. Ví dụ: `a a _ _ a b b _ c` → `a a b c`.

### Bước 2: tìm kiếm chùm tia CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Production sử dụng tiền tố cây beam search với LM fusion; Đây là bộ xương khái niệm.

### Bước 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Bước 4: inference chống lại Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Một dòng cho ASR tướng mạnh nhất vào năm 2026. Chạy trên GPU 24 GB ở ~20× thời gian thực.

### Bước 5: streaming với Vẹt đuôi dài hoặc wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR cần trạng thái encoder attention và chuyển tiếp; sử dụng thư viện hỗ trợ nó (NeMo cho Vẹt đuôi dài, `transformers` pipeline với `chunk_length_s`).

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh, ngoại tuyến, chất lượng tối đa | Thì thầm-lớn-v3-turbo |
| Đa ngôn ngữ, mạnh mẽ | Liền mạchM4T v2 |
| Streaming, độ trễ thấp | Vẹt đuôi dài-TDT-1.1B hoặc Riva |
| Edge, di động, độ trễ <500 ms | Whisper-Tiny lượng tử hóa hoặc Moonshine (2024) |
| Dạng dài | Whisper với chunking dựa trên VAD (WhisperX) |
| Miền cụ thể (y tế, pháp lý) | Fine-tune wav2vec 2.0 + hợp nhất LM miền |

## Những cạm bẫy vẫn ship vào năm 2026

- **Không có VAD.** Chạy Whisper trong im lặng tạo ra ảo giác ("Cảm ơn vì đã xem!"). Luôn cổng với VAD.
- **Ký tự so với từ so với từ phụ WER.** Báo cáo WER cấp từ * sau * chuẩn hóa (chữ thường, dấu câu bị tước bỏ).
- **Trôi ID ngôn ngữ.** LID tự động của Whisper định tuyến sai các clip nhiễu sang tiếng Nhật hoặc tiếng Wales; Buộc `language="en"` khi bạn biết.
- **Clip dài không phân đoạn.** Whisper có cửa sổ 30 giây. Sử dụng `chunk_length_s=30, stride=5` cho bất cứ thứ gì lâu hơn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-asr-picker.md`. Chọn model, chiến lược giải mã, phân đoạn và hợp nhất LM cho một mục tiêu triển khai nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó tham lam giải mã đầu ra CTC thủ công và tính toán WER dựa trên một tham chiếu.
2. **Trung bình.** Triển khai đúng cách beam search cây tiền tố ở Bước 2 (tính đến quy tắc merge trống). So sánh với tham lam trên một dataset tổng hợp 10 ví dụ.
3. **Khó.** Sử dụng `whisper-large-v3-turbo` trên [LibriSpeech test-clean](https://www.openslr.org/12). Tính toán WER trên 100 lời nói đầu tiên. So sánh với các con số đã công bố.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| CTC | token loss trống | Biên trên tất cả các căn chỉnh khung hình đến token; không phải AR. |
| RNN-T | Các streaming loss | CTC + dự đoán token tiếp theo; xử lý thứ tự từ. |
| Attention enc-tháng mười hai | Phong cách thì thầm | Encoder + decoder tham dự chéo; chất lượng ngoại tuyến tốt nhất. |
| WER | Số bạn báo cáo | `(S+D+I)/N` ở cấp độ từ. |
| Trống | Sự trống rỗng | token đặc biệt trong CTC báo hiệu "không phát ra khung này". |
| Nhiệt hạch LM | Ngôn ngữ bên ngoài model | Thêm các log-probs LM có trọng số trong quá trình beam search. |
| VAD | Cổng im lặng | Máy dò hoạt động giọng nói; cắt không phải lời nói. |

## Đọc thêm

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) - bài báo CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — bài báo RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - bài báo kinh điển năm 2022; V3-Turbo mở rộng vào năm 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) - Người dẫn đầu Bảng xếp hạng Open ASR 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) - benchmark trực tiếp trên 25+ models.
