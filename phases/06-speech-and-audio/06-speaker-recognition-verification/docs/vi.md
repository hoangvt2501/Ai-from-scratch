# Nhận dạng và xác minh người nói

> ASR hỏi "họ đã nói gì?" Nhận dạng người nói hỏi "ai đã nói điều đó?" Toán học trông giống nhau - embeddings cộng với cosin - nhưng mọi quyết định production đều phụ thuộc vào một số EER duy nhất.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ & Mel), Giai đoạn 5 · 22 (Embedding Models)
**Thời lượng:** ~45 phút

## Vấn đề

Một người dùng nói một cụm mật khẩu. Bạn muốn biết: đây có phải là người mà họ tuyên bố là (*xác minh*, 1:1), hay đó là người đầu tiên trong ngân hàng ghi danh của bạn (*nhận dạng*, 1:N)? Hoặc cả hai - đây có phải là một loa không xác định (*open-set*)?

Trước năm 2018: GMM-UBM + i-vectors. EER hợp lý nhưng dễ bị tổn thương để chuyển kênh (điện thoại so với máy tính xách tay) và cảm xúc. 2018–2022: X-vectors (Xương sống TDNN được huấn luyện với lề góc). 2022+: ECAPA-TDNN và WavLM-large embeddings. Đến năm 2026, lĩnh vực này bị chi phối bởi ba models và một số liệu.

Chỉ số là **EER** - Tỷ lệ lỗi bằng nhau. Đặt ngưỡng quyết định của bạn sao cho Tỷ lệ chấp nhận sai = Tỷ lệ từ chối sai. Sự giao nhau là EER. Được sử dụng trong mọi tờ báo, mọi bảng xếp hạng, mọi cuộc gọi mua sắm.

## Khái niệm

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Đăng ký pipeline.**: ghi lại 5–30 giây của người nói mục tiêu; tính toán embedding kích thước cố định (192-d đối với ECAPA-TDNN, 256-d đối với WavLM-large). Xác minh: nhận embedding phát biểu thử nghiệm; tính toán tương tự cosin; so sánh với một ngưỡng.

**ECAPA-TDNN (2020, vẫn chiếm ưu thế vào năm 2026).** Nhấn mạnh Attention kênh, lan truyền và tổng hợp - mạng nơ-ron trễ thời gian. Các khối conv 1D với kích thích bóp, multi-head attention gộp, tiếp theo là một lớp tuyến tính đến 192-d. Được huấn luyện về VoxCeleb 1 + 2 (2.700 người nói, 1.1 triệu lời nói) với Additive Angular Margin loss (AAM-softmax).

**WavLM-SV (2022+).** Fine-tune đường trục SSL lớn pretrained WavLM với loss AAM. Chất lượng cao hơn nhưng chậm hơn — 300+ MB so với 15 MB.

**x-vector (đường cơ sở).** TDNN + tổng hợp thống kê. Cổ điển; vẫn hữu ích trên CPU / cạnh.

**AAM-softmax.** softmax tiêu chuẩn có thêm `m` lề trong không gian góc: `cos(θ + m)` cho class chính xác. Buộc tách góc giữa các class. `m=0.2` điển hình, tỷ lệ `s=30`.

### Chấm điểm

- **Cosin** giữa đăng ký và embeddings kiểm tra. Quyết định dựa trên ngưỡng.
- **PLDA (LDA xác suất).** Dự án embeddings vào một không gian tiềm ẩn trong đó cùng một loa và loa khác có tỷ lệ likelihood dạng đóng. Thêm vào cosin để giảm +10–20% EER. Tiêu chuẩn trước năm 2020; hiện chỉ được sử dụng trong các thiết lập khép kín.
- **Chuẩn hóa điểm.** `S-norm` hoặc `AS-norm`: chuẩn hóa từng điểm số dựa trên một nhóm phương tiện mạo danh và STD. Cần thiết cho việc đánh giá tên miền chéo.

### Những con số bạn nên biết (2026)

| Model | VoxCeleb1-O EER | Tham số | Thông lượng (A100) |
|-------|-----------------|--------|-------------------|
| X-vector (Cổ điển) | 3.10% | 5 triệu | 400× RT |
| ECAPA-TDNN | 0.87% | 15 triệu | 200× RT |
| WavLM-SV lớn | 0.42% | 316 triệu | 20× RT |
| Phân đoạn Pyannote 3.1 + embedding | 0.65% | 6 triệu | 100× RT |
| Mạng ReDimNet (2024) | 0.39% | 24 triệu | 100× RT |

### Ghi chém

"Ai đã nói khi nào" trong một clip nhiều loa. Pipeline: VAD → phân đoạn → nhúng từng phân đoạn → cụm (kết tụ hoặc quang phổ) → các ranh giới trơn. stack hiện đại: `pyannote.audio` 3.1, bao gồm phân đoạn loa + embedding + phân cụm đằng sau một cuộc gọi. SOTA DER 2026 trên AMI là ~15% (giảm từ 23% vào năm 2022).

## Tự xây dựng

### Bước 1: embedding đồ chơi từ thống kê MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Không phải SOTA một dặm - chỉ để giảng dạy. `code/main.py` sử dụng điều này như một bằng chứng khái niệm về dữ liệu loa tổng hợp.

### Bước 2: độ tương tự cosin + ngưỡng

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Bước 3: EER từ các cặp tương tự

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Trả về (eer, threshold_at_eer). Báo cáo cả hai.

### Bước 4: production với SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Bước 5: diarize với pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Ứng dụng

stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Xác minh 1:1 khép kín, cạnh | ECAPA-TDNN + ngưỡng cosin |
| Xác minh bộ mở, cloud | WavLM-SV + AS-định mức |
| Diarization (cuộc họp, podcast) | `pyannote/speaker-diarization-3.1` |
| Chống giả mạo (phát lại / phát hiện deepfake) | AASIST hoặc RawNet2 |
| Nhúng nhỏ (KWS + đăng ký) | Titanet-Nhỏ (NeMo) |

## Cạm bẫy

- **Kênh không khớp.** Model được huấn luyện trên VoxCeleb (video web) ≠ âm thanh cuộc gọi điện thoại. Luôn đánh giá trên kênh mục tiêu.
- **Lời nói ngắn.** EER giảm mạnh dưới 3 giây âm thanh thử nghiệm.
- **Ghi danh với nhiễu.** Một ghi danh ồn ào đầu độc mỏ neo. Sử dụng ≥3 mẫu sạch và trung bình.
- **Ngưỡng cố định trong các điều kiện.** Luôn điều chỉnh ngưỡng trên một nhóm phát triển được giữ lại từ miền đích.
- **Cosin trên embeddings không chuẩn hóa.** L2 chuẩn hóa trước; nếu không thì độ lớn chiếm ưu thế.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-speaker-verifier.md`. Chọn model, giao thức đăng ký, kế hoạch điều chỉnh ngưỡng và các biện pháp bảo vệ chống gian lận.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xây dựng "loa" tổng hợp (các cấu hình âm khác nhau), đăng ký, tính toán EER trong danh sách dùng thử 100 cặp.
2. **Trung bình.** Sử dụng SpeechBrain ECAPA trên 30 lời nói VoxCeleb1 (5 người nói × 6 người mỗi người). Tính toán EER với cosin so với PLDA.
3. **Khó.** Xây dựng đăng ký đầy đủ → diarize → xác minh pipeline với `pyannote.audio`. Đánh giá DER trên bộ phát triển AMI.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| EER | Chỉ số dòng tiêu đề | Ngưỡng trong đó Chấp nhận sai = Từ chối sai. |
| Xác minh | 1:1 | "Đây có phải là Alice không?" |
| Nhận dạng | 1:N | "Ai đang nói?" |
| Mở bộ | Không xác định có thể | Bộ kiểm tra có thể chứa các loa chưa đăng ký. |
| Ghi danh | Đăng ký | Tính toán embedding tham chiếu của người nói. |
| AAM-softmax | Các loss | Softmax với lề góc phụ gia; buộc tách cụm. |
| PLDA | Chấm điểm cổ điển | LDA xác suất; Tỷ lệ likelihood ghi điểm trên embeddings. |
| DER | Chỉ số Diarization | Tỷ lệ lỗi Diarization - bỏ lỡ + báo động giả + nhầm lẫn. |

## Đọc thêm

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — giấy embedding sâu cổ điển.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) - Kiến trúc thống trị 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) - Đường trục SSL cho SV và diarization.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — production diarization + embedding stack.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) - bảng xếp hạng EER hiện tại trên models.
