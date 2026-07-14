# CNN và RNN cho văn bản

> Tích chập học n-gram. Tái phát ghi nhớ. Cả hai đều được thay thế bởi attention. Cả hai vẫn quan trọng trên phần cứng bị hạn chế.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 11 (PyTorch Giới thiệu), Giai đoạn 5 · 03 (Word Embeddings), Giai đoạn 4 · 02 (Tích chập từ đầu)
**Thời lượng:** ~75 phút

## Vấn đề

TF-IDF và Word2Vec đã tạo ra các vectors phẳng bỏ qua trật tự từ. Một bộ phân loại được xây dựng trên chúng không thể phân biệt `dog bites man` với `man bites dog`. Thứ tự từ đôi khi mang tín hiệu.

Hai gia đình kiến trúc đã lấp đầy khoảng trống đó trước khi transformers đến.

**Mạng tích chập cho văn bản (TextCNN).** Áp dụng tích chập 1D trên chuỗi embeddings từ. Bộ lọc có chiều rộng 3 là một trình phát hiện tam giác có thể học được: nó spans ba từ và xuất ra một điểm số. Stack các chiều rộng khác nhau (2, 3, 4, 5) để phát hiện các mẫu đa tỷ lệ. Max-pool thành một biểu diễn có kích thước cố định. Phẳng, song song, nhanh.

**Mạng lặp lại (RNN, LSTM, GRU).** Process tokens từng mạng một, duy trì trạng thái ẩn để truyền thông tin về phía trước. Tuần tự, mang bộ nhớ, độ dài đầu vào linh hoạt. Mô hình trình tự thống trị từ năm 2014 đến năm 2017, sau đó attention xảy ra.

Bài học này xây dựng cả hai, sau đó nêu tên thất bại đã thúc đẩy attention.

## Khái niệm

**TextCNN** (Kim, 2014). Tokens được nhúng. Tích chập 1D `k` chiều rộng trượt bộ lọc trên `k` gam embeddings liên tiếp, tạo ra bản đồ feature. Gộp tối đa toàn cầu trên bản đồ đó sẽ chọn kích hoạt mạnh nhất. Nối các đầu ra được gộp tối đa từ một số chiều rộng bộ lọc. Nạp vào đầu phân loại.

Tại sao nó hoạt động. Bộ lọc là một n-gram có thể học được. Max-pooling là bất biến vị trí, vì vậy "không tốt" sẽ kích hoạt cùng một feature khi bắt đầu hoặc giữa một bài đánh giá. Ba chiều rộng bộ lọc với 100 bộ lọc, mỗi bộ lọc cung cấp cho bạn 300 máy dò n-gram đã học. Training song song; không phụ thuộc tuần tự.

**RNN.** Tại mỗi bước thời gian `t`, trạng thái ẩn `h_t = f(W * x_t + U * h_{t-1} + b)`. Chia sẻ `W`, `U` `b` qua thời gian. Trạng thái ẩn tại thời điểm `T` là một bản tóm tắt của toàn bộ tiền tố. Để phân loại, hãy gộp trên `h_1 ... h_T` (tối đa, trung bình hoặc cuối cùng).

Các RNN đơn giản bị biến mất gradients. **LSTM** thêm các cổng quyết định những gì cần quên, những gì cần lưu trữ và những gì sẽ xuất ra, ổn định gradients thông qua các chuỗi dài. **GRU** đơn giản hóa LSTM thành hai cổng; hoạt động tương tự với ít parameters hơn.

**RNN hai chiều** chạy một RNN về phía trước và một RNN khác, nối các trạng thái ẩn. Mỗi đại diện của token nhìn thấy cả bối cảnh trái và phải. Cần thiết cho việc gắn thẻ các tác vụ.

```figure
rnn-unroll
```

## Tự xây dựng

### Bước 1: Nhắn tinCNN bằng PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` định hình lại `[batch, seq_len, embed_dim]` thành `[batch, embed_dim, seq_len]` vì `nn.Conv1d` coi trục giữa là kênh. Đầu ra gộp có kích thước cố định bất kể độ dài đầu vào.

### Bước 2: Bộ phân loại LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool trên trình tự, không phải pool trạng thái cuối cùng. Đối với phân loại, max-pooling thường đánh bại việc lấy trạng thái ẩn cuối cùng vì thông tin ở cuối một dãy dài có xu hướng thống trị trạng thái cuối cùng.

### Bước 3: bản demo gradient biến mất (trực giác)

Một RNN đơn giản không có cổng không thể học các phụ thuộc tầm xa. Xem xét một nhiệm vụ đồ chơi: dự đoán liệu token `A` có xuất hiện ở bất kỳ đâu trong một trình tự hay không. Nếu `A` ở vị trí 1 và dãy dài 100 tokens, gradient từ loss phải chảy ngược qua 99 phép nhân của trọng số tuần hoàn. Nếu trọng lượng nhỏ hơn 1, gradient sẽ biến mất. Nếu nhiều hơn 1, nó sẽ phát nổ.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM khắc phục điều này bằng **trạng thái ô** chạy qua mạng chỉ với các tương tác cộng (cổng quên chia tỷ lệ nó theo cách nhân lên, nhưng gradients vẫn chảy dọc theo "đường cao tốc"). GRU làm điều tương tự với ít parameters hơn. Cả hai đều cung cấp cho bạn training ổn định thông qua 100+ trình tự bước.

### Bước 4: tại sao điều này vẫn chưa đủ

Ba vấn đề vẫn tồn tại ngay cả với LSTM.

1. **Tắc nghẽn tuần tự.** Training một RNN trên một chuỗi có độ dài 1000 yêu cầu 1000 bước forward/backward nối tiếp. Không thể song song theo thời gian.
2. **vector ngữ cảnh kích thước cố định trong thiết lập encoder decoder.** decoder chỉ thấy trạng thái ẩn cuối cùng của encoder, được nén trên toàn bộ đầu vào. Đầu vào dài mất chi tiết. Bài 09 đề cập trực tiếp đến điều này.
3. **Phụ thuộc từ xa accuracy trần.** LSTM hoạt động tốt hơn RNN thông thường nhưng vẫn gặp khó khăn trong việc truyền thông tin cụ thể qua 200+ bước.

Attention đã giải quyết cả ba. Transformers đã loại bỏ hoàn toàn sự tái phát. Bài 10 là trục.

## Ứng dụng

PyTorch `nn.LSTM`, `nn.GRU` và `nn.Conv1d` đã sẵn sàng production. Mã Training là tiêu chuẩn.

Hugging Face ships pretrained embeddings bạn cắm vào làm lớp đầu vào:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Danh sách kiểm tra sử dụng khi phù hợp với ràng buộc.

- **inference cạnh / trên thiết bị.** TextCNN với GloVe embeddings nhỏ hơn 10-100 lần so với transformer. Nếu mục tiêu triển khai của bạn là điện thoại, đây là stack.
- **Streaming / phân loại trực tuyến.** RNN processes từng token một; transformers cần trình tự đầy đủ. Đối với văn bản đến theo thời gian thực, LSTM vẫn giành chiến thắng.
- **models nhỏ cho đường cơ sở.** Lặp lại nhanh một nhiệm vụ mới. Huấn luyện một TextCNN trong 5 phút trên một CPU.
- **Ghi nhãn trình tự với dữ liệu hạn chế.** BiLSTM-CRF (bài 06) vẫn là kiến trúc NER cấp production cho các câu được gắn nhãn 1k-10k.

Mọi thứ khác đều transformer.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Bài tập

1. **Dễ dàng.** Huấn luyện TextCNN trên dataset đồ chơi 3 class (bạn phát minh ra dữ liệu). Xác minh rằng chiều rộng bộ lọc (2, 3, 4) vượt trội hơn chiều rộng (3) trung bình F1.
2. **Trung bình.** Triển khai tổng thể max-pool, mean-pool và last-state pooling cho bộ phân loại LSTM. So sánh trên một dataset nhỏ; ghi lại tổng hợp nào chiến thắng và đưa ra giả thuyết tại sao.
3. **Khó.** Xây dựng một trình gắn thẻ BiLSTM-CRF NER (kết hợp bài 06 và bài này). Huấn luyện trên CoNLL-2003. So sánh với đường cơ sở chỉ CRF từ bài 06 và với một BERT fine-tune. Báo cáo thời gian, bộ nhớ và F1 của training.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Văn bảnCNN | CNN cho văn bản | Stack tích chập 1D trên embeddings từ với global max-pool. Kim (2014). |
| RNN | Mạng định kỳ | Trạng thái ẩn được cập nhật theo từng bước thời gian: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN có cổng | Thêm cổng đầu vào / quên / đầu ra + trạng thái ô. Huấn luyện ổn định qua các chuỗi dài. |
| GRU | LSTM đơn giản hơn | Hai cổng thay vì ba. Tương tự accuracy, ít parameters hơn. |
| Hai chiều | Cả hai hướng | RNN tiến + lùi được nối với nhau. Mọi token đều nhìn thấy cả hai mặt của bối cảnh của nó. |
| Biến mất gradient | Tín hiệu Training chết | Phép nhân lặp đi lặp lại với trọng số <1 trong RNN thuần túy làm cho gradients bước đầu bằng không. |

## Đọc thêm

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) - bài báo của TextCNN. Tám trang. Có thể đọc được.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - bài báo LSTM. Minh mẫn bất ngờ.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - các sơ đồ giúp mọi người có thể truy cập LSTM.
