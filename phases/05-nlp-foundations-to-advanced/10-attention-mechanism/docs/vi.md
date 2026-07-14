# Cơ chế Attention - Bước đột phá

> Người decoder ngừng nheo mắt vào một bản tóm tắt nén và bắt đầu xem toàn bộ nguồn. Mọi thứ sau này đều attention cộng với kỹ thuật.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 09 (Trình tự đến trình tự Models)
**Thời lượng:** ~45 phút

## Vấn đề

Bài học 09 kết thúc với một thất bại được đo lường. Một decoder encoder GRU được huấn luyện về nhiệm vụ sao chép đồ chơi đi từ 89% accuracy độ dài 5 đến gần cơ hội ở độ dài 80. Lý do là cấu trúc, không phải là một lỗi training: mọi thông tin mà encoder thu thập được phải phù hợp với một trạng thái ẩn có kích thước cố định và người decoder không bao giờ nhìn thấy bất cứ thứ gì khác.

Bahdanau, Cho và Bengio đã xuất bản một bản sửa lỗi ba dòng vào năm 2014. Thay vì chỉ cho decoder trạng thái encoder cuối cùng, hãy giữ mọi trạng thái encoder. Ở mỗi bước decoder, hãy tính trung bình gia quyền của encoder trạng thái trong đó trọng số cho biết "decoder cần xem xét vị trí encoder `i` bao nhiêu ngay bây giờ?" Trung bình gia quyền đó là bối cảnh và nó thay đổi mỗi decoder bước.

Đó là toàn bộ ý tưởng. Transformers mở rộng nó. Self-attention áp dụng nó cho một trình tự duy nhất. Multi-head attention chạy song song. Nhưng phiên bản 2014 đã phá vỡ nút thắt cổ chai và một khi bạn có nó, chuyển hướng sang transformers là kỹ thuật chứ không phải khái niệm.

## Khái niệm

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

Ở mỗi bước decoder `t`:

1. Sử dụng `s_{t-1}` trạng thái ẩn decoder trước đó làm **truy vấn**.
2. Ghi điểm nó dựa trên mọi encoder trạng thái ẩn `h_1, ..., h_T`. Một vô hướng cho mỗi encoder vị trí.
3. Softmax điểm số để có trọng số attention `α_{t,1}, ..., α_{t,T}` tổng đó là 1.
4. Bối cảnh vector `c_t = Σ α_{t,i} * h_i`. Trung bình gia quyền của encoder tiểu bang.
5. Decoder lấy `c_t` cộng với token đầu ra trước đó, tạo ra token tiếp theo.

Trung bình gia quyền là điểm. Khi decoder cần dịch "Je" thành "I", nó đặt trọng số trạng thái encoder hơn "Je" cao và các trạng thái khác thấp. Khi nó cần "không", nó có trọng lượng "pas" cao. Bối cảnh vector định hình lại từng bước.

## Hình dạng (thứ cắn tất cả mọi người)

Đây là nơi mọi attention triển khai đều sai ngay lần đầu tiên. Đọc chậm.

| Chuyện | Hình dạng | Ghi chú |
|-------|-------|-------|
| Encoder trạng thái ẩn `H` | `(T_enc, d_h)` | Nếu BiLSTM, `d_h = 2 * d_hidden` |
| Decoder trạng thái ẩn `s_{t-1}` | `(d_s,)` | Một vector |
| Attention điểm `e_{t,i}` | vô hướng | Một mỗi encoder vị trí |
| Trọng lượng Attention `α_{t,i}` | vô hướng | Sau khi softmax qua tất cả `i` |
| Bối cảnh vector `c_t` | `(d_h,)` | Hình dạng giống như trạng thái encoder |

**Điểm Bahdanau (cộng thêm).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` có hình dạng `(d_s,)`, `h_i` có hình dạng `(d_h,)`.
- `W_a` có hình dạng `(d_attn, d_s)`. `U_a` có hình dạng `(d_attn, d_h)`.
- Tổng của chúng bên trong tanh có hình dạng `(d_attn,)`.
- `v_α` có hình dạng `(d_attn,)`. Tích bên trong có `v_α` sụp đổ thành vô hướng. **Đây là những gì `v_α` làm.** Nó không phải là phép thuật. Đó là phép chiếu biến một vector attention mờ thành một điểm vô hướng.

**Điểm Lương (nhân).** Ba biến thể:

- `dot`: `e_{t,i} = s_t^T * h_i`. Yêu cầu `d_s == d_h`. Ràng buộc cứng. Bỏ qua nếu encoder của bạn là hai chiều.
- `general`: `e_{t,i} = s_t^T * W * h_i` với hình dạng `W` `(d_s, d_h)`. Loại bỏ ràng buộc bằng độ mờ.
- `concat`: về cơ bản là dạng Bahdanau. Hiếm khi được sử dụng vì hai cái đầu tiên rẻ hơn.

**Một Bahdanau / Lương đáng để đặt tên.** Bahdanau sử dụng `s_{t-1}` (trạng thái decoder * trước * tạo ra từ hiện tại). Lương sử dụng `s_t` (trạng thái *sau*). Việc trộn lẫn chúng với nhau sẽ tạo ra các gradients sai một cách tinh tế mà cực kỳ khó gỡ lỗi. Chọn một tờ giấy và tuân theo quy ước của nó.

```figure
attention-heatmap
```

## Tự xây dựng

### Bước 1: phụ gia (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Kiểm tra hình dạng của bạn so với bảng trên. `encoder_states` có hình dạng `(T_enc, d_h)`. `projected_enc` có hình dạng `(T_enc, d_attn)`. `projected_dec` có hình dạng `(d_attn,)` và phát sóng. `combined` có hình dạng `(T_enc, d_attn)`. `scores` có hình dạng `(T_enc,)`. `weights` có hình dạng `(T_enc,)`. `context` có hình dạng `(d_h,)`. Ship nó.

### Bước 2: Hàm chấm và tổng quát

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Ba dòng mỗi dòng. Đây là lý do tại sao tờ báo của Lương hạ cánh. Tương tự accuracy trên hầu hết các tác vụ, ít mã hơn rất nhiều.

### Bước 3: một ví dụ số đã làm việc

Cho ba trạng thái encoder (khoảng "cat", "sat", "mat") và trạng thái decoder phù hợp nhất với trạng thái đầu tiên, phân phối attention tập trung vào vị trí 0. Nếu trạng thái decoder thay đổi để căn chỉnh với trạng thái cuối cùng, attention sẽ chuyển sang vị trí 2. Bối cảnh vector các bản nhạc.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Hàng đầu tiên thắng. Sau đó, di chuyển trạng thái decoder gần hơn với trạng thái encoder thứ ba và quan sát trọng số dịch chuyển. Đó là nó. Attention là alignment rõ ràng.

### Bước 4: tại sao đây là cầu nối đến transformers

Dịch ngôn ngữ trên sang Q/K/V:

- **Truy vấn** = decoder trạng thái `s_{t-1}`
- **Chìa khóa** = encoder trạng thái (những gì chúng tôi ghi điểm)
- **Giá trị **= encoder trạng thái (trọng số và tổng của chúng ta)

Trong attention cổ điển, khóa và giá trị là cùng một thứ. Self-attention tách chúng: bạn có thể truy vấn một trình tự chống lại chính nó, với các phép chiếu đã học khác nhau cho K và V. Multi-head attention chạy nó song song với các phép chiếu đã học khác nhau. Transformers stack toàn bộ giai đoạn nhiều lần và thả RNN.

Toán học cũng vậy. Các hình dạng giống nhau. Bước nhảy vọt sư phạm từ Bahdanau attention sang attention chấm-product theo tỷ lệ chủ yếu là ký hiệu.

## Ứng dụng

PyTorch và TensorFlow ship attention trực tiếp.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Đó là một lớp transformer attention. Truy vấn batch của 5 vị trí, key/value batch của 10 vị trí, 128 độ mờ mỗi vị trí, 8 đầu. `output` là các truy vấn tăng cường ngữ cảnh mới. `weights` là ma trận alignment 5x10 mà bạn có thể hình dung.

### Khi attention cổ điển vẫn quan trọng

- Sư phạm. Phiên bản một đầu, một lớp, dựa trên RNN làm cho mọi khái niệm có thể nhìn thấy được.
- Các tác vụ trình tự trên thiết bị mà transformers không phù hợp.
- Bất kỳ bài báo nào từ năm 2014-2017. Bạn sẽ đọc sai nó mà không biết quy ước của Bahdanau.
- Phân tích alignment chi tiết trong MT. Trọng lượng attention thô là một công cụ có thể giải thích ngay cả trên transformer models và việc đọc chúng đòi hỏi phải biết chúng là gì.

### Bẫy attention trọng lượng như lời giải thích

Trọng lượng Attention trông có thể giải thích được. Chúng là trọng số tổng bằng một giữa các vị trí; bạn có thể âm mưu cho chúng; cao có nghĩa là "đã nhìn vào điều này". Người đánh giá yêu thích chúng.

Chúng không dễ giải thích như vẻ ngoài của chúng. Jain và Wallace (2019) đã chỉ ra rằng các phân phối attention có thể được hoán vị và thay thế bằng các lựa chọn thay thế tùy ý mà không thay đổi dự đoán model cho một số nhiệm vụ. Không bao giờ báo cáo trọng lượng attention làm bằng chứng về lý luận mà không có kiểm tra cắt bỏ hoặc phản thực tế.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Bài tập

1. **Dễ dàng.** Thực hiện mặt nạ `softmax` để đệm tokens trong encoder attention trọng lượng bằng không. Thử nghiệm trên batch có trình tự có độ dài thay đổi.
2. **Trung bình.** Thêm multi-head attention vào dạng `general` Luong. Chia `d_h` thành `n_heads` nhóm, chạy attention trên mỗi đầu, nối với nhau. Xác minh trường hợp một đầu khớp với cách triển khai trước đó của bạn.
3. **Khó.** Huấn luyện một decoder encoder GRU với Bahdanau attention làm nhiệm vụ sao chép đồ chơi từ bài 09. Cốt truyện accuracy so với độ dài trình tự. So sánh với đường cơ sở không attention. Bạn sẽ thấy khoảng cách mở rộng khi chiều dài tăng lên, xác nhận attention dỡ bỏ nút thắt cổ chai.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Attention | Nhìn vào mọi thứ | Trung bình gia quyền của một chuỗi giá trị, trọng số được tính từ sự tương đồng của khóa truy vấn. |
| Truy vấn, khóa, giá trị | QKV | Ba phép chiếu: Q hỏi, K là cái gì phù hợp, V là cái gì trả về. |
| Phụ gia attention | Bahdanau | Điểm chuyển tiếp: `v^T tanh(W q + U k)`. |
| attention nhân | Lương chấm / tướng | Điểm số là `q^T k` hoặc `q^T W k`. Rẻ hơn, accuracy giống nhau trên hầu hết các tác vụ. |
| Ma trận Alignment | Bức tranh đẹp | Attention trọng số dưới dạng lưới `(T_dec, T_enc)`. Đọc nó để xem model đã tham gia vào những gì. |

## Đọc thêm

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - tờ giấy.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) - ba biến thể điểm số và so sánh của chúng.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) - cảnh báo về khả năng diễn giải.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) - hướng dẫn có thể chạy với PyTorch.
