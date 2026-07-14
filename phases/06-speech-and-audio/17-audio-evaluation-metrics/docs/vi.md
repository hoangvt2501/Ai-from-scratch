# Đánh giá âm thanh - WER, MOS, UTMOS, MMAU, FAD và Bảng xếp hạng mở

> Bạn không thể ship những gì bạn không thể đo lường. Bài học này đặt tên cho các chỉ số năm 2026 cho mọi tác vụ âm thanh: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), ngôn ngữ âm thanh (MMAU, LongAudioBench), âm nhạc (FAD, CLAP) và loa (EER). Cộng với bảng xếp hạng nơi bạn so sánh.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 04, 06, 07, 09, 10; Giai đoạn 2 · 09 (Đánh giá Model)
**Thời lượng:** ~60 phút

## Vấn đề

Mỗi tác vụ âm thanh có nhiều chỉ số, mỗi chỉ số đo một trục khác nhau. Sử dụng sai số liệu là cách bạn ship một model trông tuyệt vời trên bảng điều khiển của mình và rất production. Danh sách kinh điển năm 2026:

| Nhiệm vụ | Tiểu học | Trung học |
|------|---------|-----------|
| ASR | WER | CER · RTFx · Độ trễ token đầu tiên |
| TTS | MOS / UTMOS | GIÂY · WER-on-ASR-khứ hồi · CER · TTFA |
| Nhân bản giọng nói | SECS (ECAPA cosine) | MOS · CER |
| Xác minh người nói | EER | minDCF · FAR / FRR tại điểm vận hành |
| Ghi chém | DER | JER · nhầm lẫn người nói |
| Phân loại âm thanh | TOP-1 · mAP | F1 macro · mỗi class recall |
| Tạo âm nhạc | THỜI TRANG | Vỗ tay · bảng điều khiển nghe MOS |
| Ngôn ngữ âm thanh model | MMAU-Chuyên nghiệp | Băng ghế dài · AudioCaps FENSE |
| Streaming S2S | Độ trễ P50/P95 | WER · MOS |

## Khái niệm

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR chỉ số

**WER (Tỷ lệ lỗi từ).** `(S + D + I) / N`. Chữ thường, dải dấu câu, chuẩn hóa số trước khi chấm điểm. Sử dụng `jiwer` hoặc `whisper_normalizer` của OpenAI. &lt; 5% = giọng nói đọc chẵn lẻ của con người.

**CER (Tỷ lệ lỗi ký tự).** Cùng một công thức, cấp ký tự. Được sử dụng cho các ngôn ngữ thanh điệu (Quan Thoại, Quảng Đông) trong đó phân đoạn từ không rõ ràng.

**RTFx (hệ số thời gian thực nghịch đảo).** Số giây âm thanh được xử lý trên mỗi giây đồng hồ treo tường. Cao hơn là tốt hơn. Vẹt đuôi dài-TDT đạt 3380×. Whisper-large-v3 là ~30×.

**Độ trễ token đầu tiên.** Đồng hồ treo tường từ đầu vào âm thanh đến token bản ghi đầu tiên. Rất quan trọng đối với streaming. Deepgram Nova-3: ~150 mili giây.

### TTS chỉ số

**MOS (Điểm ý kiến trung bình).** Đánh giá của con người 1-5. Tiêu chuẩn vàng nhưng chậm. Thu thập 20+ người nghe mỗi mẫu, 100+ mẫu mỗi model.

**UTMOS (2022-2026).** Đã học công cụ dự đoán MOS. Tương quan ~0,9 với MOS của con người trên benchmarks tiêu chuẩn. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Tương tự Encoder Cosin của loa).** Để sao chép giọng nói. ECAPA embedding cosin giữa đầu ra tham chiếu và đầu ra nhân bản. &gt; 0,75 = bản sao có thể nhận biết.

**WER-on-ASR-round-trip.** Chạy Whisper qua đầu ra TTS, tính toán WER so với văn bản đầu vào. Bắt hồi quy về độ hiểu. SOTA năm 2026: &lt; 2% CER.

**TTFA (thời gian đến âm thanh đầu tiên).** Độ trễ đồng hồ treo tường. Kokoro-82M: ~100 mili giây; F5-TTS: ~1 giây.

### Nhân bản giọng nói cụ thể

**SECS + MOS + CER **dưới dạng bộ ba. Nhân bản đạt điểm SECS cao nhưng MOS thấp có nghĩa là âm sắc đúng nhưng không tự nhiên; ngược lại có nghĩa là giọng nói tự nhiên nhưng nói sai.

### Xác minh người nói

**EER (Tỷ lệ lỗi bằng nhau).** Ngưỡng trong đó Tỷ lệ chấp nhận sai bằng Tỷ lệ từ chối sai. ECAPA trên VoxCeleb1-O: 0,87%.

**minDCF (chi phí phát hiện tối thiểu).** Chi phí có trọng số tại điểm vận hành đã chọn (thường là FAR=0,01). Phù hợp với production hơn EER.

### Ghi chém

**DER (Tỷ lệ lỗi Diarization).** `(FA + Miss + Confusion) / total_speaker_time`. Bài phát biểu bị bỏ lỡ + bài phát biểu báo động giả + nhầm lẫn với người nói, mỗi bài phát biểu là một phân số. Các cuộc họp AMI: DER ~10-20% là thực tế. pyannote 3.1 + Precision-2 thương mại: &lt;10% DER trên âm thanh được ghi tốt.

**JER (Tỷ lệ lỗi Jaccard).** Thay thế cho DER, mạnh mẽ đến bias phân đoạn ngắn.

### Phân loại âm thanh

Đa nhãn: **mAP (trung bình Precision)** trên tất cả classes. AudioSet: 0.548 mAP cho BEATs-iter3.

Độc quyền nhiều class: **top-1, top-5 accuracy**. Lệnh giọng nói v2: 99.0% top-1 (Audio-MAE).

Mất cân bằng: **macro F1 **+ **mỗi class recall**. Báo cáo mỗi class — tổng hợp accuracy ẩn classes thất bại.

### Tạo âm nhạc

**FAD (Khoảng cách âm thanh Fréchet).** Khoảng cách giữa các phân phối VGGish-embedding của âm thanh thực và âm thanh được tạo. MusicGen-small trên MusicCaps: 4.5. Âm nhạcLM: 4.0. Thấp hơn, tốt hơn.

**Điểm CLAP.** Điểm alignment văn bản-âm thanh sử dụng embeddings CLAP. &gt; 0,3 = alignment hợp lý.

**Bảng điều khiển nghe MOS.** Vẫn là lời cuối cùng cho âm nhạc tiêu dùng. Suno v5 ELO 1293 trên TTS Arena (từ sở thích của con người được ghép nối).

### benchmarks ngôn ngữ âm thanh

**MMAU (Hiểu biết đa âm thanh lớn).** 10k cặp âm thanh-QA.

**MMAU-Pro.** 1800 mục cứng, bốn danh mục: giọng nói / âm thanh / âm nhạc / đa âm thanh. Cơ hội ngẫu nhiên 25% trên 4 chiều. Gemini 2.5 Pro tổng thể ~60%; đa âm thanh ~22% trên tất cả các models.

**LongAudioBench.** Clip dài nhiều phút với các truy vấn ngữ nghĩa. Âm thanh Flamingo Next đánh bại Gemini 2.5 Pro.

**AudioCaps / Clotho.** Phụ đề benchmarks. Các chỉ số SPICE, CIDEr, FENSE.

### Streaming Chuyển giọng nói thành giọng nói

**Độ trễ P50 / P95 / P99.** Đồng hồ treo tường từ giọng nói cuối của người dùng đến phản hồi âm thanh đầu tiên. Moshi: 200 mili giây; GPT-4o Thời gian thực: 300 ms.

**WER / MOS** trên đầu ra.

**Khả năng phản hồi của sà lan.** Thời gian từ khi người dùng ngắt đến khi trợ lý tắt tiếng. Mục tiêu &lt; 150 ms.

### Bảng xếp hạng năm 2026

| Bảng xếp hạng | Bài hát | URL |
|------------|--------|-----|
| Mở Bảng xếp hạng ASR (HF) | Tiếng Anh + đa ngôn ngữ + dạng dài | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| Đấu trường TTS (HF) | Tiếng Anh TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Bài phát biểu phân tích nhân tạo | TTS + STT, ELO từ các phiếu bầu được ghép đôi | `artificialanalysis.ai/speech` |
| MMAU-Chuyên nghiệp | Lý luận LALM | `mmaubenchmark.github.io` |
| Băng ghế loa / VoxSRC | Nhận dạng loa | `voxsrc.github.io` |
| Tập hợp con nhạc MMAU | Âm nhạc LALM | (trong MMAU) |
| NGHE benchmark | Âm thanh tự giám sát | `hearbenchmark.com` |

## Tự xây dựng

### Bước 1: WER với chuẩn hóa

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Bước 2: TTS WER khứ hồi

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Bước 3: SECS để sao chép giọng nói

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Bước 4: FAD để tạo nhạc

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Bước 5: EER để xác minh người nói (cùng mã với Bài 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Ứng dụng

Ghép nối mọi triển khai với harness đánh giá cố định chạy trên mỗi bản cập nhật model. Ba quy tắc cơ bản:

1. **Chuẩn hóa trước khi chấm điểm.** Chữ thường, dải dấu câu, mở rộng số. Báo cáo quy tắc chuẩn hóa.
2. **Báo cáo phân phối, không phải mức trung bình.** P50/P95/P99 độ trễ. Mỗi class recall để phân loại. Mỗi danh mục cho MMAU.
3. **Chạy một benchmark công khai chính tắc.** Ngay cả khi dữ liệu production của bạn khác nhau, báo cáo trên Open ASR / TTS Arena / MMAU cho phép người đánh giá so sánh táo với táo.

## Cạm bẫy

- **Ngoại suy UTMOS.** Được huấn luyện về giọng nói rõ ràng theo phong cách VCTK; điểm âm thanh ồn ào / nhân bản / cảm xúc kém.
- **Bảng điều khiển MOS bias.** 20 Amazon Mechanical Turk workers ≠ 20 người dùng mục tiêu. Trả tiền cho một bảng điều khiển tên miền nếu tiền đặt cược cao.
- **FAD phụ thuộc vào tập tham chiếu.** So sánh với cùng một phân phối tham chiếu trên models.
- **WER tổng hợp.** Tổng thể 5% WER có thể ẩn 30% WER trên giọng nói có dấu. Báo cáo theo lát nhân khẩu học.
- **Độ bão hòa benchmark công cộng.** Hầu hết các models biên giới đều ở gần trần trên benchmarks tiêu chuẩn. Xây dựng một nhóm nội bộ phản ánh lưu lượng truy cập của bạn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-audio-evaluator.md`. Chọn chỉ số, benchmarks và định dạng báo cáo cho bất kỳ bản phát hành model âm thanh nào.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Tính toán WER / CER / EER / SECS / FAD-ish / MMAU-ish trên đầu vào đồ chơi.
2. **Trung bình.** Xây dựng một harness WER khứ hồi TTS. Chạy đầu ra Kokoro hoặc F5-TTS của bạn thông qua Whisper. Tính toán WER trên 50 prompts. Gắn cờ prompts với WER &gt; 10%.
3. **Khó.** Chấm điểm lựa chọn LALM Bài 10 của bạn trên giọng nói MMAU-Pro + nhiều tập hợp con âm thanh (mỗi tập 50 mục). Báo cáo từng danh mục accuracy và so sánh với số lượng đã xuất bản.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| WER | ASR điểm | `(S+D+I)/N` ở cấp độ từ sau khi chuẩn hóa. |
| CER | Nhân vật WER | Đối với ngôn ngữ giai điệu hoặc hệ thống cấp char. |
| MOS | Ý kiến con người | Đánh giá 1-5; 20+ người nghe × 100 mẫu. |
| UTMOS | ML công cụ dự đoán MOS | Học model; tương quan ~0,9 với MOS của con người. |
| GIÂY | Sự tương đồng giữa giọng nói và bản sao | ECAPA cosin giữa tham chiếu và nhân bản. |
| EER | Điểm verif của loa | Ngưỡng trong đó FAR = FRR. |
| DER | Điểm Diarization | (FA + Bỏ lỡ + Nhầm lẫn) / tổng. |
| THỜI TRANG | Chất lượng thế hệ âm nhạc | Khoảng cách Fréchet trên VGGish embeddings. |
| RTFx | Thông lượng | Giây âm thanh trên mỗi giây đồng hồ treo tường. |

## Đọc thêm

- [jiwer](https://github.com/jitsi/jiwer) — WER/CER thư viện với các tiện ích chuẩn hóa.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) - đã học dự đoán MOS.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — tiêu chuẩn thế hệ âm nhạc.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) - Bảng xếp hạng trực tiếp năm 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — bảng xếp hạng TTS bình chọn của con người.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) - Bảng xếp hạng lý luận LALM.
- [HEAR benchmark](https://hearbenchmark.com/) - benchmarks SSL âm thanh.
