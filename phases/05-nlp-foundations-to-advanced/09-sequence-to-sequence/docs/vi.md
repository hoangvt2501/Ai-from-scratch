# Trình tự đến trình tự Models

> Hai RNN giả vờ là một phiên dịch. Nút thắt cổ chai mà họ gặp phải là lý do attention tồn tại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 08 (CNN + RNN cho văn bản), Giai đoạn 3 · 11 (PyTorch Giới thiệu)
**Thời lượng:** ~75 phút

## Vấn đề

Phân loại ánh xạ một chuỗi có độ dài thay đổi thành một nhãn duy nhất. Bản dịch ánh xạ một chuỗi có độ dài thay đổi với một chuỗi có độ dài thay đổi khác. Đầu vào và đầu ra tồn tại trong các từ vựng khác nhau, có thể là các ngôn ngữ khác nhau, không đảm bảo về độ dài ngang nhau.

Kiến trúc seq2seq (Sutskever, Vinyals, Le, 2014) đã bẻ khóa điều này bằng một công thức đơn giản có chủ ý. Hai RNN. Người ta đọc câu nguồn và tạo ra một vector ngữ cảnh có kích thước cố định. Cái kia đọc vector đó và tạo ra câu đích token token. Cùng một mã bạn đã viết cho bài 08, dán lại với nhau theo cách khác nhau.

Điều này đáng để nghiên cứu vì hai lý do. Thứ nhất, nút thắt cổ chai vector ngữ cảnh là thất bại hữu ích nhất về mặt sư phạm trong NLP. Nó thúc đẩy mọi thứ attention và transformers giỏi. Thứ hai, công thức training (ép giáo viên, sampling theo lịch trình beam search inference) vẫn áp dụng cho mọi hệ thống thế hệ hiện đại bao gồm cả LLMs.

## Khái niệm

**Encoder.** Một RNN đọc câu nguồn. Trạng thái ẩn cuối cùng của nó là **context vector** — một bản tóm tắt kích thước cố định của toàn bộ đầu vào. Không mất gì ngoài nguồn, được cho là vậy.

**Decoder.** Một RNN khác được khởi tạo từ ngữ cảnh vector. Ở mỗi bước, nó lấy token đã tạo trước đó làm đầu vào và tạo ra sự phân phối trên từ vựng đích. Sample hoặc argmax để chọn token tiếp theo. Nạp lại. Lặp lại cho đến khi tạo ra `<EOS>` token hoặc đạt đến chiều dài tối đa.

**Training:** loss entropy chéo ở mỗi bước decoder, được tổng hợp theo trình tự. Backprop tiêu chuẩn qua thời gian thông qua cả hai mạng.

**Giáo viên cưỡng bức.** Trong quá trình training, đầu vào của decoder ở bước `t` là * sự thật cơ bản * token ở vị trí `t-1`, không phải dự đoán trước đó của chính decoder. Điều này ổn định training; Nếu không có nó, những sai lầm ban đầu sẽ đổ xuống và model không bao giờ học được. Tại inference, bạn phải sử dụng dự đoán riêng của model, vì vậy luôn có khoảng cách phân phối train/inference. Khoảng cách đó được gọi là **phơi nhiễm bias**.

**Nút thắt cổ chai.** Mọi thứ encoder học được về nguồn phải được ép vào một bối cảnh đó vector. Câu dài mất chi tiết. Những từ hiếm bị mờ. Sắp xếp lại thứ tự (chat noir so với mèo đen) phải được ghi nhớ chứ không phải tính toán.

Attention (bài 10) khắc phục điều này bằng cách để decoder nhìn vào trạng thái ẩn *mỗi* encoder, không chỉ trạng thái cuối cùng. Đó là toàn bộ sân cỏ.

```figure
lstm-gates
```

## Tự xây dựng

### Bước 1: encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` có hình dạng `[batch, seq_len, hidden_dim]` - một trạng thái ẩn cho mỗi vị trí đầu vào. `hidden` có hình dạng `[1, batch, hidden_dim]` - bước cuối cùng. Bài 08 nói "gộp trên đầu ra để phân loại". Ở đây chúng ta giữ trạng thái ẩn cuối cùng làm ngữ cảnh vector và bỏ qua các đầu ra trên mỗi bước.

### Bước 2: decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder được gọi từng bước một. Đầu vào: một batch của một tokens và trạng thái ẩn hiện tại. Đầu ra: logits từ vựng cho token tiếp theo và trạng thái ẩn được cập nhật.

### Bước 3: training vòng lặp với giáo viên ép buộc

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Hai núm vặn đáng để đặt tên. `ignore_index=0` bỏ qua loss trên tokens đệm. `teacher_forcing_ratio` là xác suất sử dụng token thực so với dự đoán của model ở mỗi bước. Bắt đầu từ 1.0 (ép giáo viên đầy đủ) và ủ xuống ~0.5 trên training để thu hẹp khoảng cách bias phơi nhiễm.

### Bước 4: inference vòng lặp (tham lam)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Giải mã tham lam chọn token xác suất cao nhất ở mỗi bước. Nó có thể đi lang thang: một khi bạn commit đến một token, bạn không thể bỏ lại nó. **Beam search** giữ cho các chuỗi phần `k` trên cùng tồn tại và chọn chuỗi hoàn chỉnh có điểm cao nhất ở cuối. Chiều rộng chùm tia 3-5 là tiêu chuẩn.

### Bước 5: nút thắt cổ chai, được chứng minh

Huấn luyện model về nhiệm vụ sao chép đồ chơi: `[a, b, c, d, e]` nguồn, `[a, b, c, d, e]` mục tiêu. Tăng độ dài trình tự. Quan sát accuracy.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Một trạng thái ẩn GRU duy nhất không thể ghi nhớ đầu vào 40 token một cách dễ dàng. Thông tin có ở mọi bước encoder, nhưng decoder chỉ nhìn thấy trạng thái cuối cùng. Attention khắc phục vấn đề này trực tiếp.

## Ứng dụng

PyTorch có các mẫu seq2seq dựa trên `nn.Transformer` và dựa trên `nn.LSTM`. Thư viện `transformers` của Hugging Face ships đầy đủ encoder-decoder models (BART, T5, mBART, NLLB) được huấn luyện trên hàng tỷ tokens.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Các decoders encoder hiện đại đã loại bỏ RNN cho transformers. Hình dạng cấp cao (encoder, decoder, generate-token-by-token) giống hệt với bài báo seq2seq năm 2014. Cơ chế bên trong mỗi khối là khác nhau.

### Khi nào vẫn nên tiếp cận seq2seq dựa trên RNN

Hầu như không bao giờ, đối với các dự án mới. Các trường hợp ngoại lệ cụ thể:

- Streaming dịch mà bạn sử dụng đầu vào từng token một với bộ nhớ giới hạn.
- Tạo văn bản trên thiết bị, trong đó chi phí bộ nhớ transformer quá cao.
- Sư phạm. Hiểu được nút thắt cổ chai encoder-decoder là con đường nhanh nhất để hiểu lý do tại sao transformers chiến thắng.

### bias phơi nhiễm và các biện pháp giảm thiểu

- **sampling theo lịch trình.** Tỷ lệ ép giáo viên ủ trong quá trình training để model học cách phục hồi sau những sai lầm của chính mình.
- **Rủi ro tối thiểu training.** Huấn luyện về điểm BLEU cấp câu thay vì entropy chéo cấp token. Gần hơn với những gì bạn thực sự muốn.
- **Học tăng cường fine-tuning.** Thưởng cho trình tạo trình tự bằng một số liệu. Được sử dụng trong LLM RLHF hiện đại.

Cả ba vẫn áp dụng cho thế hệ dựa trên transformer.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Bài tập

1. **Dễ dàng.** Thực hiện nhiệm vụ sao chép đồ chơi. Huấn luyện GRU seq2seq trên các cặp đầu vào-đầu ra trong đó mục tiêu bằng nguồn. Đo accuracy ở độ dài 5, 10, 20. Tái tạo nút thắt cổ chai.
2. **Trung bình.** Thêm giải mã beam search với chiều rộng chùm tia 3. Đo BLEU trên một kho dữ liệu song song nhỏ chống lại lòng tham. Ghi lại nơi beam search chiến thắng (thường là cuối cùng tokens) và nơi nào không có sự khác biệt.
3. **Khó.** Fine-tune `facebook/bart-base` trên một dataset diễn giải 10k cặp. So sánh đầu ra chùm-4 của fine-tuned model với đầu ra model cơ sở trên đầu vào bị giữ. Báo cáo BLEU và chọn 10 ví dụ định tính.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Encoder | Đầu vào RNN | Đọc nguồn. Tạo trạng thái ẩn trên mỗi bước và vector ngữ cảnh cuối cùng. |
| Decoder | Đầu ra RNN | Khởi tạo từ ngữ cảnh vector. Tạo từng tokens mục tiêu một. |
| Bối cảnh vector | Tóm tắt | Trạng thái cuối cùng encoder ẩn. Kích thước cố định. Nút thắt cổ chai attention giải quyết. |
| Giáo viên ép buộc | Sử dụng tokens thực sự | Cung cấp sự thật cơ bản token trước đó vào training lúc. Ổn định học tập. |
| Tiếp xúc bias | Train/test khoảng cách | Model được huấn luyện trên tokens thực sự không bao giờ thực hành phục hồi từ những sai lầm của chính mình. |
| Beam search | Giải mã tốt hơn | Giữ cho top-k chuỗi từng phần sống động ở mỗi bước thay vì tham lam cam kết. |

## Đọc thêm

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — bài báo seq2seq gốc. Bốn trang.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) - giới thiệu GRU và khung decoder encoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - tờ báo attention. Đọc ngay sau bài học này.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — mã seq2seq + attention có thể xây dựng.
