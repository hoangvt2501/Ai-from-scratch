# Giải mã suy đoán và EAGLE-3

> Giai đoạn 7 · Bài học 16 đã chứng minh toán học: quy tắc từ chối Leviathan bảo toàn chính xác phân phối của trình xác minh. Bài học này là quan điểm training stack của năm 2026 production giải mã đầu cơ. EAGLE-3 đã biến model dự thảo từ một xấp xỉ rẻ tiền thành một mạng nhỏ được xây dựng có mục đích được huấn luyện trên các trạng thái ẩn của chính người xác minh, sau đó thêm một vòng lặp kiểm tra training thời gian để căn chỉnh các phân phối huấn luyện và inference của nó. Kết quả: Tăng tốc đầu cuối từ 3× đến 6,5×, tỷ lệ mỗi token được chấp nhận trên 0,9 khi trò chuyện, không đánh đổi phân phối. Mỗi production inference stack vào năm 2026 đều ships nó theo mặc định.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 7 · 16 (toán học giải mã suy đoán), Giai đoạn 10 · 12 (tối ưu hóa inference)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Nêu định lý Leviathan trong một câu và chứng minh rằng vòng lặp suy đoán tạo ra các mẫu được phân phối giống hệt nhau cho người xác minh.
- Đi bộ theo tiến trình hai năm từ giải mã thông số kỹ thuật vani (Leviathan 2023) đến EAGLE, EAGLE-2 và EAGLE-3 và đặt tên cho giới hạn chính xác mà mỗi bước đã loại bỏ.
- Tính toán tốc độ dự kiến từ tỷ lệ chấp nhận `α` và tỷ lệ chi phí từ bản nháp đến người xác minh `c`, đồng thời chọn `N` độ dài bản nháp tối ưu cho từng chế độ.
- Thực hiện vòng lặp đầu cơ đầy đủ từ đầu: nháp, xác minh, từ chối mẫu từ phần dư, quay lại KV cache khi bị từ chối, phát ra token thưởng khi được chấp nhận hoàn toàn.

## Vấn đề

Giải mã tự hồi quy trên model 70B chạy với tốc độ có thể 35 tokens mỗi giây trên H100. GPU không ở đâu gần bão hòa. Băng thông bộ nhớ là trần: mỗi token tải 70B trọng số từ HBM, thực hiện một bước số học và tạo ra một float. Các đơn vị tính toán hầu như không hoạt động.

Giải mã đầu cơ biến điều đó thành một vấn đề thông lượng mà bạn thực sự có thể giải quyết. Một bản nháp rẻ tiền đề xuất `N` tokens trong `N` lần chuyển tiếp nhỏ. Trình xác minh chạy một lần trên tiền tố cộng với tất cả các bản nháp `N`. Nếu phân phối của người xác minh tại vị trí `i` đồng ý với bản nháp (theo nghĩa thống kê, chúng tôi sẽ làm chính xác), chúng tôi chấp nhận; nếu không, chúng tôi từ chối và lấy mẫu hiệu chỉnh từ phân phối dư. Một chuyển tiếp model lớn duy nhất tạo ra tối đa `N+1` tokens được chấp nhận thay vì một.

Định lý quan trọng là Leviathan, Kalman, Matias (ICML 2023): phân phối đầu ra giống hệt với những gì sampling từ trình xác minh trực tiếp sẽ tạo ra. Không gần đúng. Giống hệt nhau. Đây là toàn bộ lý do giải mã đầu cơ được chấp nhận trong production - đó là một tối ưu hóa độ trễ thuần túy mà không có sự đánh đổi chất lượng.

Giai đoạn 7 · Bài học 16 cho bạn là toán học. Những gì bài học này mang lại cho bạn là training stack. Một bản nháp tốt có giá trị tăng tốc gấp 2× so với bản nháp giá rẻ. EAGLE, EAGLE-2 và EAGLE-3 (Li và cộng sự, 2024–2025) đã biến "bản nháp = phiên bản nhỏ hơn của cùng một model" thành một ngành kỹ thuật chính xác. 2026 production inference servers mặc định là EAGLE-3.

## Khái niệm

### Bất biến: Leviathan từ chối sampling

Hãy để `p(t)` là bản phân phối của bản nháp cho token tiếp theo với một số tiền tố và `q(t)` là của người xác minh. Lấy mẫu một token `d ~ p` nháp. Chấp nhận với xác suất `min(1, q(d) / p(d))`. Khi từ chối, lấy mẫu từ `(q - p)_+ / ||(q - p)_+||_1` phân phối dư. Các mẫu kết quả được phân phối theo `q`. Điều này đúng bất kể `p` tệ như thế nào - nó càng tồi tệ, bạn càng từ chối thường xuyên, nhưng đầu ra vẫn chính xác.

Stack `N` trong số các cuộc gọi này liên tiếp bằng cách sử dụng một forward pass xác minh trên `prefix + d_1 + ... + d_N`. Trình xác minh trả về `q_1, q_2, ..., q_{N+1}` đồng thời. Đi từ trái sang phải. Trong lần từ chối đầu tiên ở vị trí `j`, lấy mẫu từ `residual(q_j, p_j)` và dừng lại. Khi được chấp nhận hoàn toàn, lấy mẫu một token thưởng từ `q_{N+1}`.

### Điều gì quyết định tăng tốc

Giả sử `α` là tỷ lệ chấp nhận dự kiến cho mỗi token được soạn thảo. Giả sử `c = cost(draft) / cost(verifier)` là tỷ lệ chi phí. Số lượng tokens được chấp nhận dự kiến cho mỗi người xác minh chuyển tiếp là:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Tổng thời gian tường dự kiến cho mỗi token được chấp nhận là `(N * c + 1) / E[accepted]`. Giảm thiểu điều đó đối với `N` và bạn sẽ có được điểm ngọt ngào. Đối với `α = 0.8, c = 0.05`: `N` tối ưu là khoảng 5–7, tốc độ là 3.2×. Đối với `α = 0.95, c = 0.02`: `N` tối ưu là khoảng 8–10, tăng tốc đẩy 5×.

Đòn bẩy lớn nhất là `α`. Đi từ `α = 0.6` (bản nháp vani) đến `α = 0.9` (EAGLE-3) ở `N = 5` cố định sẽ đưa bạn từ 2,2 tokens được chấp nhận dự kiến cho mỗi trình xác minh lên 4,1. Thông lượng nhiều hơn gần 2× từ cùng một trình xác minh.

### Tiến trình hai năm

**Đầu cơ vani (Leviathan, 2023).** Draft model là một LLM nhỏ hơn được huấn luyện độc lập từ cùng một gia đình. Dễ dàng nối dây, `α ≈ 0.6`, tăng tốc tốt nhất khoảng 2×.

**EAGLE-1 (Li và cộng sự, 2024).** Bản nháp là một transformer nhỏ - thường là một hoặc hai lớp - lấy trạng thái ẩn lớp cuối cùng của người xác minh làm đầu vào và dự đoán trực tiếp token tiếp theo. Bởi vì bản nháp nhìn thấy đại diện feature của người xác minh, nên sự phân phối của nó gần với người xác minh hơn nhiều. `α` tăng lên 0,7–0,8.

**EAGLE-2 (Li và cộng sự, 2024).** Thêm một cây nháp động: thay vì đề xuất một chuỗi `N` tokens duy nhất, hãy đề xuất một cây ứng cử viên nhỏ, chấm điểm mỗi người với người xác minh trong một forward pass (attention cây) và đi theo con đường có xác suất cao nhất. Độ dài bản nháp trở nên thích ứng trên mỗi bước. `α` trên mỗi đường dẫn được chấp nhận token leo lên trên 0,85.

**EAGLE-3 (Li và cộng sự, 2025, NeurIPS).** Hai thay đổi nữa. Đầu tiên, loại bỏ hoàn toàn dự đoán feature loss - EAGLE-1/2 huấn luyện bản nháp để khớp với trạng thái ẩn của người xác minh, giới hạn lượng dữ liệu giúp ích. EAGLE-3 huấn luyện trực tiếp trên dự đoán token. Thứ hai, kiểm tra thời gian training (TTT): trong training dự thảo, cung cấp lại các dự đoán trước đó của bản nháp dưới dạng đầu vào qua nhiều bước, giống như cách nó hoạt động ở inference. Điều này căn chỉnh phân phối huấn luyện và kiểm tra và ngăn chặn tích lũy lỗi. Tốc độ đo lường: lên đến 6,5× khi trò chuyện, cải thiện 38% thông lượng ở batch 64 trong SGLang trên H100.

### KV cache rollback

Xác minh mở rộng KV cache của trình xác minh bằng cách `N` mục nhập trong một lần. Nếu từ chối xảy ra ở vị trí `j`, nội dung bộ nhớ đệm `j-1` vị trí trước đây hiện sai. Hai cách triển khai phổ biến: ghi vào bộ đệm cào và commit khi chấp nhận (vLLM, TensorRT-LLM) hoặc giữ KV cache vật lý cộng với độ dài logic và cắt bớt khi từ chối. Dù bằng cách nào, chi phí rollback là byte trên mỗi lớp trên đầu, không đáng kể bên cạnh chi phí chuyển tiếp.

Đối với tìm kiếm cây EAGLE-2, trình xác minh chạy attention với mặt nạ không nhân quả tôn trọng cấu trúc liên kết cây. Kỹ thuật phức tạp nhưng tính toán là một cuộc gọi flash-attention tiêu chuẩn với mặt nạ tùy chỉnh.

### Dự thảo kiến trúc vào năm 2026

| Chiến lược | Loại bản nháp | `α` | Tăng tốc | Chi phí Training |
|----------|-----------|-----|---------|---------------|
| Vani | Riêng biệt LLM nhỏ | 0.55-0.70 | 1.8-2.3× | Không có (tái sử dụng model nhỏ hiện có) |
| Medusa | Đầu LM bổ sung trên trình xác minh | 0.65-0.75 | 2-3× | ~1 tỷ SFT tokens |
| ĐẠI BÀNG-1 | transformer 1 lớp trên các trạng thái ẩn | 0.70-0.80 | 2.5-3× | ~60 tỷ tokens |
| ĐẠI BÀNG-2 | EAGLE-1 + cây nháp động | 0.80-0.88 | 3-4× | ~60 tỷ tokens |
| ĐẠI BÀNG-3 | Nhiệt hạch feature nhiều lớp + TTT | 0.88-0.92 | 3,5-6,5× | ~60-200 tỷ tokens |
| Lookahead | Không có bản nháp (lặp lại Jacobi) | N/A | 1.3-1.6× | Không có |

Vào năm 2026 production: vLLM và SGLang mặc định là EAGLE-3 khi có sẵn, EAGLE-2 nếu không. TensorRT-LLM có đường dẫn Medusa nhanh nhất cho Meta và NVIDIA models công khai. llama.cpp ships bản nháp vani để triển khai CPU.

## Tự xây dựng

Xem `code/main.py`. Đây là vòng lặp suy đoán Leviathan đầy đủ với tất cả các mảnh: nháp N, vượt qua song song của người xác minh, từ chối mỗi vị trí, sampling dư, token thưởng, rollback KV và xác minh thực nghiệm rằng phân phối đầu ra khớp với sampling trực tiếp từ `q`.

### Bước 1: quy tắc từ chối

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Bước 2: phân phối dư

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Bước 3: một bước đầu cơ đầy đủ

Hàm `spec_step` nháp `N` tokens từ `p`, sau đó xác minh tất cả chúng trong một đánh giá `q` song song. Đối với mỗi token nháp, nó áp dụng quy tắc từ chối và trong lần từ chối đầu tiên, nó lấy mẫu hiệu chỉnh từ phần dư. Nếu mọi thứ chấp nhận, nó sẽ phát ra một token thưởng từ `q_{N+1}`.

### Bước 4: Sổ sách kế toán rollback KV

Trình mô phỏng theo dõi một `kv_length` logic trên mỗi worker. Khi chấp nhận các bản nháp `k`, `kv_length += k`. Khi bị từ chối ở vị trí `j`, bộ nhớ đệm đã được ghi quá `j`, nhưng độ dài logic được đặt thành `prefix_length + j + 1` - một sau token hiệu chỉnh. Các lần đọc tiếp theo sẽ cắt bớt độ dài logic.

### Bước 5: Kiểm tra Leviathan

Chạy 50.000 bước suy đoán. Đếm phân phối thực nghiệm của tokens được chấp nhận. So sánh với 50.000 mẫu trực tiếp từ `q`. Thống kê chi bình phương phải thấp hơn giá trị tới hạn. Định lý vượt qua trong thực tế.

### Bước 6: tăng tốc so với α

Quét chất lượng bản nháp bằng cách làm nhiễu `p` ra khỏi `q` ở các biên độ khác nhau. Đo `α`, sau đó vẽ biểu đồ tokens dự kiến cho mỗi cuộc gọi của trình xác minh dưới dạng hàm của `α` và `N`. Mã in một bảng cho thấy cách chất lượng bản nháp EAGLE-3-class (`α ≈ 0.9`) mở khóa 4–5 tokens cho mỗi cuộc gọi của người xác minh.

## Ứng dụng

`vllm serve` cấp Production với EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang với EAGLE-3 ở batch 64 trên H100: thông lượng cao hơn khoảng 1,38× so với giải mã vani batch-64, theo bài báo EAGLE-3.

Khi nào cần giải mã đầu cơ:

- Bất kỳ khối lượng công việc trò chuyện tương tác nào mà độ trễ p50 quan trọng hơn thông lượng cao nhất.
- Tạo mã và đầu ra có cấu trúc (JSON, SQL). `α` trên 0,9 vì phân phối mục tiêu rất dễ dự đoán.
- Thế hệ dạng dài (hàng nghìn tokens). Tốc độ khấu hao tiếp tục trả tiền.

Khi nào không:

- models rất nhỏ (< 3B). Bản nháp không rẻ hơn nhiều so với trình xác minh.
- Triển khai CPU batch-1 nhỏ. Chi phí bộ nhớ của model dự thảo có thể không đáng giá.
- Sáng tạo temperature rất cao sampling nơi `α` sụp đổ.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-eagle3-tuner.md`. Với khối lượng công việc inference (model, kích thước batch, độ trễ mục tiêu, hồ sơ tác vụ), nó đề xuất một chiến lược giải mã suy đoán và điều chỉnh parameters (họ nháp, `N`, độ sâu cây, chuyển đổi nhận biết temperature).

## Bài tập

1. Chạy `code/main.py`. Xác nhận số liệu thống kê chi bình phương trên kiểm tra phân phối Leviathan vẫn dưới giá trị tới hạn 95% trên 50.000 mẫu.

2. Quét `N` từ 1 đến 10 với `α` được giữ ở 0,9 và `c` được giữ ở 0,04. Vẽ biểu đồ dự kiến tokens cho mỗi cuộc gọi của người xác minh và thời gian tường thực tế trên mỗi token. Tìm `N` giảm thiểu thời gian tường. Giải thích hình dạng của đường cong.

3. Sửa đổi mã để mô phỏng tìm kiếm cây EAGLE-2: ở mỗi bước, bản nháp đề xuất một cây có hình dạng `[2, 2, 2]` (tám đường dẫn ứng cử viên). Trình xác minh chạy một lần và đường dẫn được chấp nhận có xác suất cao nhất sẽ thắng. Tính toán `α` trên mỗi lá và tổng tokens cho mỗi lệnh gọi trình xác minh. So sánh với giải mã thông số kỹ thuật chuỗi tuyến tính ở điện toán tương đương.

4. Triển khai trình mô phỏng KV rollback hàng loạt cho hai trình tự đồng thời. Trình tự A đã chấp nhận tất cả các bản nháp; trình tự B từ chối ở vị trí 2. Cho thấy rằng `kv_length` chính xác được cập nhật cho mỗi trình tự và không có công việc nào bị lãng phí.

5. Đọc Phần 4 (Thử nghiệm Training thời gian của bài báo EAGLE-3). Giải thích trong hai câu tại sao training nháp ngây thơ không có TTT lại bị phơi nhiễm bias và tại sao việc cung cấp dự đoán của chính bản nháp trong training khắc phục nó. Kết nối điều này với tài liệu sampling theo lịch trình trong seq2seq.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Quy tắc Leviathan | "tối thiểu (1, q trên p)" | Bernoulli accept/reject với xác suất `min(1, q(d)/p(d))`, bảo toàn phân phối trình xác minh chính xác khi bạn lấy mẫu từ phần dư khi bị từ chối |
| Phân phối dư | "(q trừ p) cộng, chuẩn hóa" | `(q - p)_+` kẹp ở số không và chuẩn hóa lại - phân phối chính xác cho mẫu từ khi loại bỏ |
| Tỷ lệ chấp nhận α | "Tần suất bản nháp đúng" | Xác suất thành công của Bernoulli mỗi token dự kiến theo quy tắc từ chối; chi phối tất cả các phép toán tăng tốc |
| ĐẠI BÀNG-1 | "Dự thảo trạng thái ẩn" | Bản nháp transformer nhỏ có điều kiện dựa trên trạng thái ẩn lớp cuối cùng của người xác minh (Li và cộng sự, 2024) |
| ĐẠI BÀNG-2 | "Cây nháp động" | EAGLE-1 cộng với một cây tiếp tục ứng cử viên được ghi điểm với attention cây trong một lần vượt qua trình xác minh |
| ĐẠI BÀNG-3 | "Kiểm tra training lần" | Giảm loss dự đoán feature, huấn luyện dự đoán token trực tiếp với dự thảo được cung cấp đầu ra của chính nó trong training |
| Kiểm tra Training lần (TTT) | "phơi nhiễm bias khắc phục" | Chạy bản nháp tự hồi quy trong training để huấn luyện và kiểm tra các phân phối đầu vào phù hợp - tương tự trực tiếp của sampling đã lên lịch |
| KV rollback | "Hoàn tác bản nháp bị từ chối" | Sổ sách kế toán đặt lại KV cache của người xác minh về độ dài tiền tố được chấp nhận sau khi bị từ chối |
| Tiền thưởng token | "Người tự do" | Khi tất cả các bản nháp `N` chấp nhận, hãy lấy mẫu thêm một bản `q_{N+1}` mà không mất thêm chi phí xác minh |
| Cây attention | "Xác minh nhiều ứng viên cùng một lúc" | Attention với mặt nạ không nhân quả tôn trọng cấu trúc liên kết của cây nháp; tính toán `q_i` cho mọi nút trong cây trong một forward pass |

## Đọc thêm

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — bài báo cơ bản và định lý tương đương
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) - giới thiệu độc lập đồng thời với bằng chứng rõ ràng
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) - EAGLE-1, bản nháp có điều kiện trạng thái ẩn
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — tìm kiếm cây động
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — mặc định production năm 2026
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) - cách tiếp cận thay thế không có bản nháp
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo production chuẩn với tất cả các chiến lược được kết nối
