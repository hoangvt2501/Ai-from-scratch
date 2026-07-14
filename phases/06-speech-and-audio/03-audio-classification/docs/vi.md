# Phân loại âm thanh - Từ k-NN trên MFCC đến AST và BEAT

> Tất cả mọi thứ từ "tiếng chó sủa vs còi báo động" đến "đây là ngôn ngữ nào" đều là phân loại âm thanh. Các features là mels. Kiến trúc di chuyển mỗi thập kỷ. Đánh giá vẫn là AUC, F1 và mỗi class recall.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 6 · 02 (Quang phổ & Mel), Giai đoạn 3 · 06 (CNN), Giai đoạn 5 · 08 (CNN & RNN cho văn bản)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn nhận được một clip dài 10 giây. Bạn muốn biết: "nó là gì?" Âm thanh đô thị (còi báo động, máy khoan, chó), lệnh nói (yes/no/stop), ID ngôn ngữ (en/es/ar), cảm xúc của người nói (angry/neutral) hoặc âm thanh môi trường (indoor/outdoor, bập bẹ). Tất cả những điều này đều là *phân loại âm thanh*, và vào năm 2026, kiến trúc cơ bản đã trưởng thành: log-mel → CNN hoặc Transformer → softmax.

Khó khăn cốt lõi không phải là mạng. Đó là dữ liệu. datasets âm thanh có class imbalance tàn bạo, dịch chuyển miền mạnh (sạch và nhiễu) và nhiễu nhãn (ai quyết định "tiếng bập bẹ đô thị" so với "nhiễu nhà hàng"?). 80% vấn đề là quản lý, tăng cường và đánh giá, không hoán đổi CNN lấy Transformer.

## Khái niệm

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN trên MFCC (đường cơ sở những năm 1990).** Làm phẳng MFCC trên mỗi clip, tính toán sự tương đồng cosin với một ngân hàng được dán nhãn, trả lại đa số phiếu bầu của K hàng đầu. Mạnh mẽ một cách đáng ngạc nhiên về datasets sạch, nhỏ (Lệnh giọng nói, ESC-50). Chạy không GPU.

**2D CNN về log-mels (2015-2019).** Coi log-mel `(T, n_mels)` như một hình ảnh. Áp dụng ResNet-18 hoặc kiểu VGG. Gộp trung bình toàn cầu trục thời gian. Softmax hơn classes. Vẫn là cơ sở trong hầu hết các cuộc thi kaggle năm 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Vá lỗi log-mel (ví dụ: 16×16 bản vá), thêm vị trí embeddings, nạp vào ViT. Trạng thái nghệ thuật trên AudioSet (mAP 0.485) để học có giám sát.

**BEATs và WavLM-base (2024-2026).** Tự giám sát pretraining trên hàng triệu giờ. Fine-tune thực hiện nhiệm vụ của mình với 1-10% dữ liệu được giám sát mà bạn cần. Vào năm 2026, đây là điểm bắt đầu mặc định cho âm thanh không phải giọng nói. BEATs-iter3 đánh bại AST 1-2 mAP trên AudioSet trong khi sử dụng 1/4 máy tính.

**Whisper-encoder như một xương sống đóng băng (2024).** Lấy encoder của Whisper, thả decoder, đính kèm bộ phân loại tuyến tính. Gần SOTA trên ID ngôn ngữ và phân loại sự kiện đơn giản mà không cần tăng cường âm thanh. Đường cơ sở "bữa trưa miễn phí".

### Class imbalance là thách thức thực sự

ESC-50: 50 classes, 40 clip mỗi clip - cân bằng, dễ dàng. UrbanSound8K: 10 classes, mất cân bằng 10: 1. Bộ âm thanh: 632 classes với đuôi dài 100.000: 1. Các kỹ thuật hoạt động:

- Cân bằng sampling trong quá trình training (không được đánh giá).
- Mixup: nội suy tuyến tính hai clip (và nhãn của chúng) dưới dạng tăng cường.
- SpecAugment: che các dải tần và thời gian ngẫu nhiên. Đơn giản; quan trọng.

### Đánh giá

- Độc quyền đa lớp (Lệnh giọng nói): accuracy top 1, accuracy top 5.
- Đa nhãn đa lớp (AudioSet, UrbanSound-style): precision trung bình trung bình (mAP).
- Mất cân bằng nặng: mỗi class recall + F1 macro.

Những con số năm 2026 bạn nên biết:

| Benchmark | Đường cơ sở | SOTA 2026 | Nguồn |
|-----------|----------|-----------|--------|
| ESC-50 · | 82% (AST) | 97,0% (BEATs-iter3) | Bài báo BEATs (2024) |
| Bộ âm thanh mAP | 0,485 (AST) | 0.548 (BEATs-iter3) | Bảng xếp hạng HEAR 2026 |
| Lệnh giọng nói v2 | 98% (CNN) | 99.0% (Âm thanh-MAE) | NGHE kết quả phiên bản 2 |

## Tự xây dựng

### Bước 1: làm nổi bật

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Bước 2: tóm tắt độ dài cố định

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Đơn giản nhưng mạnh mẽ: trung bình + variance theo thời gian cho embedding cố định 26 mờ cho MFCC 13 coef. Chạy ngay lập tức. Đánh bại các đường cơ sở NN hiện đại trên ESC-50 gần đây nhất là vào năm 2017.

### Bước 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Bước 4: nâng cấp lên CNN trên log-mels

Trong PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M parameters. Huấn luyện trong ~10 phút trên ESC-50 với một RTX 4090 duy nhất. 80%+ accuracy.

### Bước 5: mặc định năm 2026 - fine-tune BEAT

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Đối với BEAT, hãy sử dụng `microsoft/BEATs-base` thông qua thư viện `beats`; transformers API có hình dạng giống nhau.

## Ứng dụng

stack năm 2026:

| Tình huống | Bắt đầu với |
|-----------|-----------|
| dataset tí hon (<1000 clip) | k-NN trên MFCC có nghĩa là (đường cơ sở của bạn) + tăng cường âm thanh |
| dataset trung bình (1K–100K) | BEAT hoặc AST fine-tune |
| dataset lớn (>100K) | Huấn luyện từ đầu hoặc fine-tune Whisper-encoder |
| Thời gian thực, cạnh | 40-MFCC CNN, lượng tử hóa thành int8 (kiểu KWS) |
| Đa nhãn (AudioSet) | BEATs-iter3 với BCE loss + mixup + SpecAugment |
| ID ngôn ngữ | MMS-LID, SpeechBrain VoxLingua107 đường cơ sở |

Quy tắc quyết định: **bắt đầu với xương sống đóng băng, không phải model mới**. Fine-tuning đầu BEATs giúp bạn có được 95% SOTA trong vài giờ, không phải vài tuần.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-classifier-designer.md`. Chọn kiến trúc, tăng cường, chiến lược cân bằng class và chỉ số đánh giá cho một nhiệm vụ phân loại âm thanh nhất định.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Nó huấn luyện đường cơ sở k-NN MFCC trên dataset tổng hợp 4 class (âm thanh thuần túy ở các cao độ khác nhau). Báo cáo ma trận nhầm lẫn.
2. **Trung bình.** Thay thế `summarize` bằng [trung bình, var, lệch, kẹp]. Nhịp gộp 4 khoảnh khắc có mean+var trên cùng một dataset tổng hợp không?
3. **Khó.** Sử dụng `torchaudio`, huấn luyện CNN 2D trên ESC-50 gấp 1. Báo cáo accuracy xác thực chéo gấp 5 lần. Thêm SpecAugment (mặt nạ thời gian = 20, mặt nạ tần số = 10) và báo cáo delta.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bộ âm thanh | ImageNet của âm thanh | 2 triệu clip của Google, 632 class được dán nhãn yếu trên YouTube dataset. |
| ESC-50 · | Phân loại nhỏ benchmark | 50 classes × 40 clip âm thanh môi trường. |
| AST | Quang phổ âm thanh Transformer | ViT trên các bản vá log-mel; SOTA năm 2021. |
| NHỊP ĐẬP | Âm thanh tự giám sát | Microsoft model, iter3 dẫn đầu AudioSet kể từ năm 2026. |
| Trộn lẫn | Tăng cường cặp | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| Tăng cường thông số | Tăng cường dựa trên mặt nạ | Zero-out thời gian ngẫu nhiên và dải tần số của quang phổ. |
| mAP | Chỉ số đa nhãn chính | precision trung bình trung bình trên các classes và ngưỡng. |

## Đọc thêm

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) - kiến trúc kỷ lục từ năm 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) - mặc định 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) - tăng cường âm thanh chủ đạo.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) - 50-class benchmark vẫn tồn tại.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) - Phân loại YouTube 632-class; vẫn là bản vị vàng.
