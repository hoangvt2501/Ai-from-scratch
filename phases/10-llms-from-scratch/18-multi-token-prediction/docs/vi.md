# Dự đoán đa Token (MTP)

> Mỗi LLM tự hồi quy từ GPT-2 đến Llama 3 đoàn tàu trên một loss cho mỗi vị trí: dự đoán token tiếp theo. DeepSeek-V3 đã thêm một loss thứ hai cho mỗi vị trí: dự đoán token sau đó. 14B parameters bổ sung (trên model 671B) đã được chưng cất trở lại model chính thông qua gradient luồng và các đầu MTP được huấn luyện đã được tái sử dụng tại inference làm máy soạn thảo giải mã đầu cơ với 80% + chấp nhận. Thông lượng thế hệ 1,8× được miễn phí. Bài học này xây dựng mô-đun MTP tuần tự từ báo cáo kỹ thuật DeepSeek, tính toán loss và bố cục parameter đầu chia sẻ, đồng thời giải thích lý do tại sao MTP giữ chuỗi nhân quả trong khi MTP song song ban đầu của Gloeckle và cộng sự đã phá vỡ nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 10 · 04 (training trước một GPT mini), Giai đoạn 10 · 15 (giải mã suy đoán)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Nêu mục tiêu training MTP và suy ra loss chung trên các độ sâu dự đoán.
- Giải thích sự khác biệt giữa các đầu MTP song song của Gloeckle et al. (2024) và các mô-đun MTP tuần tự của DeepSeek-V3 và lý do tại sao thiết kế tuần tự bảo toàn chuỗi nhân quả.
- Tính toán parameter và chi phí bộ nhớ của việc thêm mô-đun MTP vào quá trình chạy trước training.
- Triển khai một mô-đun MTP từ đầu: embedding chia sẻ, khối transformer trên mỗi độ sâu, phép chiếu và đầu ra được chia sẻ.

## Vấn đề

Dự đoán token tiếp theo là tiêu chuẩn LLM training mục tiêu. Mọi trạng thái ẩn đều được giám sát để dự đoán chính xác một điều: ngay sau đó token. Đó là một tín hiệu yếu đáng ngạc nhiên. Hầu hết thông tin trong một chuỗi vượt ra ngoài một token - cấu trúc, tính mạch lạc, tính thực tế, dòng số học. Người model phải học những điều đó bằng cách tích lũy nhiều tín hiệu một token trên hàng nghìn tỷ tokens.

MTP đặt câu hỏi: điều gì sẽ xảy ra nếu mọi trạng thái ẩn được giám sát để dự đoán nhiều tokens trong tương lai cùng một lúc? Gloeckle et al. (Meta, 2024) cho thấy điều này hữu ích. Việc triển khai của họ đặt một số đầu ra độc lập lên trên xương sống, mỗi đầu dự đoán một độ lệch khác nhau. Song song, đơn giản, nhưng các đầu nhìn thấy cùng một trạng thái ẩn mà không có bất kỳ sự tinh chỉnh phân cấp nào - và các dự đoán không xâu chuỗi về mặt nhân quả, vì vậy chúng không thể được sử dụng để giải mã suy đoán.

DeepSeek-V3 (tháng 12 năm 2024) đã thiết kế lại MTP dưới dạng các mô-đun tuần tự giữ chuỗi nhân quả ở mỗi độ sâu dự đoán. model dự đoán `t+1` từ `h_i^(0)`, sau đó dự đoán `t+2` từ một trạng thái ẩn mới `h_i^(1)` kết hợp `h_i^(0)` với `E(t+1)` embedding, v.v. Mỗi độ sâu là khối transformer nhỏ của riêng nó. Đầu embedding được chia sẻ và đầu ra được chia sẻ giữ parameter mức khiêm tốn. Ở quy mô của DeepSeek-V3, 14B parameters bổ sung trên các mô-đun MTP trên trọng lượng model chính 671B. Chi phí 2% đó đã mua các tín hiệu training đặc hơn VÀ một bản nháp giải mã đầu cơ được tạo sẵn ở inference.

Bài học này xây dựng một mô-đun MTP duy nhất và loss độ sâu D từ đầu. Toán học rất gọn gàng. Việc triển khai là 150 dòng.

## Khái niệm

### Công thức MTP tuần tự

DeepSeek-V3 thêm `D` mô-đun MTP trên đầu model chính. Mỗi mô-đun `k` (cho `k = 1..D`) dự đoán token ở độ sâu `k` - nghĩa là `t_{i+k}` được đặt tiền tố thông qua vị trí `i`.

`k` mô-đun bao gồm:

- Một khối transformer `T_k` với attention và MLP riêng.
- Một ma trận chiếu `M_k` kết hợp trạng thái ẩn độ sâu trước đó với embedding của token chân lý nền sâu tiếp theo.
- embedding `E` dùng chung (giống như model chính).
- Đầu ra được chia sẻ `Out` (giống như model chính).

Ở training, đối với tiền tố thông qua vị trí `i`, trạng thái ẩn trên mỗi độ sâu là:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Dự đoán cho mỗi độ sâu là:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

loss mỗi chiều sâu là entropy chéo so với `t_{i+k}` chân lý cơ bản:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Khớp nối loss qua các độ sâu:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` là một hệ số trọng số nhỏ - DeepSeek-V3 sử dụng 0,3 cho 10% đầu tiên của training và 0,1 sau đó. Tổng training loss là `L_main + L_MTP`.

### Tại sao tuần tự, không song song

MTP song song ban đầu của Gloeckle có đầu ra D, mỗi đầu được áp dụng trực tiếp cho `h_i^(0)`. Mỗi đầu dự đoán `t_{i+k}` từ cùng một trạng thái ẩn xương sống. Điều đó huấn luyện tốt, nhưng các dự đoán không có điều kiện cho nhau. Bạn không thể sử dụng đầu ra của `head_1` để giúp `head_2` - các đầu bắn song song.

Thiết kế tuần tự của DeepSeek-V3 xây dựng `h_i^(k)` từ `h_i^(k-1)` cộng với token embedding `E(t_{i+k})` tiếp theo thực tế. Điều đó bảo tồn chuỗi nhân quả: để dự đoán `t_{i+k+1}`, mô-đun ở độ sâu `k+1` nhìn thấy những gì ở `t_{i+k}`. Điều này có cấu trúc giống hệt với cách một decoder tự hồi quy tiêu thụ đầu ra của chính nó - làm cho các mô-đun MTP có thể được sử dụng trực tiếp như các bản soạn thảo giải mã suy đoán.

Tại inference: đưa `h_i^(k-1)` và `t_{i+k}` đã nháp vào `k+1` mô-đun, nhận dự đoán cho `t_{i+k+1}`. Lặp lại. Đó chính xác là một bản nháp kiểu EAGLE, sử dụng mô-đun MTP được huấn luyện làm mạng nháp. DeepSeek-V3 báo cáo chấp nhận 80%+ trên mô-đun MTP đầu tiên và tăng tốc ~1,8×.

### Kế toán Parameter

Đối với một model có `h` và từ vựng ẩn `V`:

- model chính: hàng tỷ parameters, cộng với một đầu ra có kích thước `V * h`.
- Đầu ra được chia sẻ: sử dụng lại đầu của model chính. Không có tham số bổ sung.
- Shared embedding: sử dụng lại embedding của model chính. Không có tham số thừa.
- Mỗi mô-đun MTP:
  - `M_k` chiếu: `(2h) * h = 2h^2`.
  - `T_k` khối Transformer: attention (`4h^2` cho MHA) cộng với MLP (thường `8h^2` cho SwiGLU có tỷ lệ 8/3). Khoảng `12h^2` mỗi khối.

Tổng số dư thừa trên mỗi mô-đun: `~14h^2`. Đối với `h = 7168` của DeepSeek-V3, D = 1 mô-đun: `~14 * 7168^2 = ~720M` parameters trên giấy. DeepSeek-V3 báo cáo 14B - sự khác biệt chủ yếu là các lớp chuyên gia cũng được MoE trong mô-đun MTP.

### Phần thưởng giải mã đầu cơ

Trong quá trình training trước, các mô-đun MTP làm chậm training khoảng 10% (tính toán chuyển tiếp nhiều hơn, thêm loss). Phần thưởng gấp đôi:

1. Tín hiệu training dày đặc hơn. Mỗi trạng thái ẩn nhìn thấy các mục tiêu giám sát D+1. Hiệu ứng đo lường đối với MMLU, GSM8K, MATH, HumanEval: cải thiện một vài điểm phần trăm nhất quán trong các vết cắt bỏ của DeepSeek-V3.

2. Bản nháp giải mã đầu cơ miễn phí tại inference. Mô-đun MTP đã được huấn luyện để dự đoán vài tokens tới. Được tái sử dụng như một mạng nháp, nó cung cấp tỷ lệ chấp nhận 80%+. Ở cấp độ đó, giải mã thông số kỹ thuật N=3 hoặc N=5 cho thông lượng 1,8×. Chi phí 10% training lần trả lại trong lần đầu tiên bạn chạy inference.

### Quan hệ với EAGLE

EAGLE huấn luyện một model nháp nhỏ RIÊNG BIỆT sau khi training trước. MTP nướng bản nháp vào training trước. Hai cách tiếp cận hội tụ về tỷ lệ chấp nhận tương tự nhau nhưng thông qua các pipelines khác nhau:

| Kích thước | ĐẠI BÀNG-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| Khi được huấn luyện | Sau khi training | Trong quá trình training trước |
| Tương thích ngược với trọng lượng hiện có | Có | Không (cần huấn luyện lại) |
| Tham số nháp | 1-2 lớp transformer | 1 khối transformer + chiếu |
| Tỷ lệ chấp nhận | 0.88-0.92 | 0,80+ ở độ sâu 1 |
| Lợi ích ngoài việc tăng tốc | Chỉ giải mã suy đoán | Tín hiệu training dày đặc hơn + tăng tốc |

## Tự xây dựng

`code/main.py` xây dựng một mô-đun MTP duy nhất từ đầu đến cuối: embedding chia sẻ, phép chiếu, khối transformer, đầu ra được chia sẻ. Sau đó, nó tính toán loss entropy chéo trên mỗi độ sâu trên một chuỗi tổng hợp ngắn và in số lượng parameter theo thành phần. Từ vựng đồ chơi 32 tokens giữ cho các con số có thể đọc được.

### Bước 1: bảng embedding dùng chung

Một bảng `vocab_size x hidden` duy nhất được sử dụng bởi model chính VÀ bởi mọi mô-đun MTP ở mọi độ sâu. Không phải là bản sao thứ hai - theo nghĩa đen là cùng một tensor.

### Bước 2: kết hợp mỗi chiều sâu

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Real DeepSeek-V3 nối hai vectors RMSNormed với `[2h]` và dự án với ma trận `h x 2h`. Đồ chơi sử dụng vector bổ sung để stdlib ngắn gọn.

### Bước 3: khối transformer ở độ sâu k

Self-attention cộng với MLP. Trong đồ chơi, một khối attention tuyến tính một lớp và MLP SwiGLU giữ cho cấu trúc có thể nhìn thấy mà không cần numpy.

### Bước 4: đầu ra được chia sẻ

Sử dụng lại phép chiếu đầu ra của model chính. Logits qua từ vựng.

### Bước 5: mỗi loss độ sâu

Entropy chéo của softmax (logits) so với sự thật cơ bản token ở `k` bù đắp. Tổng hợp qua các độ sâu với hệ số tỷ lệ `lambda / D`.

### Bước 6: parameter kế toán

In tổng số parameter, số lượng chia sẻ (embedding, đầu) và số lượng bổ sung trên mỗi mô-đun. Hiển thị tỷ lệ MTP bổ sung so với kích thước model chính.

## Ứng dụng

MTP được tích hợp vào DeepSeek-V3 (Tháng 12 năm 2024) và dòng DeepSeek-R1. Tại inference:

- stack phục vụ riêng của DeepSeek sử dụng các mô-đun MTP như một decoders đầu cơ.
- vLLM và SGLang có đường dẫn tích hợp cho DeepSeek-V3 MTP kể từ tháng 4 năm 2026.
- Hướng dẫn ROCm SGLang của AMD hiển thị một config giải mã suy đoán MTP cụ thể với tốc độ đo được 1,8× trên checkpoint V3.

Khi nào nên sử dụng MTP trong lần chạy trước training mới:

- Bạn kiểm soát toàn bộ tiền training pipeline và muốn lưu trữ tín hiệu training dày đặc hơn.
- Bạn biết rằng bạn sẽ phục vụ model trên quy mô lớn và muốn giải mã đầu cơ miễn phí.
- Kích thước ẩn của bạn ít nhất là 4096. Ở tỷ lệ 1B, chi phí trên bị tổn thương nhiều hơn mức tăng giúp ích.

Khi nào không:

- Fine-tuning một model dày đặc được huấn luyện trước hiện có. Mô-đun MTP không được huấn luyện.
- Nghiên cứu models nơi bạn muốn một đường cơ sở rõ ràng để so sánh. MTP thay đổi kiến trúc.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mtp-planner.md`. Với thông số kỹ thuật chạy trước khi training (kích thước model, dữ liệu, tính toán), nó trả về một kế hoạch để tích hợp MTP: số độ sâu D, lịch trình `lambda`, chi phí bộ nhớ và hệ thống dây giải mã suy đoán thời gian inference.

## Bài tập

1. Chạy `code/main.py`. Hiển thị mỗi độ sâu loss giảm đơn điệu khi tín hiệu tổng hợp tăng cường. Sửa đổi tổng hợp để sử dụng một mẫu cố định và xác minh cả tổn thất độ sâu 1 và độ sâu 2 hội tụ.

2. Tính toán chi phí parameter cho một model 70B dày đặc (8192, 80 lớp ẩn) với mô-đun D = 1 MTP. So sánh với chi phí chung 3B được báo cáo của DeepSeek-V14. Giải thích lý do tại sao con số của DeepSeek cao hơn: khối MTP transformer kế thừa cùng một cấu trúc MoE, làm tăng số lượng parameter trên mỗi mô-đun.

3. Triển khai D = 2 trong đồ chơi: thêm mô-đun MTP thứ hai lấy h ^ (1) và dự đoán `t_{i+2}`. Xác minh loss chung và kế toán parameter khớp với phương trình 19-21 của bài báo DeepSeek.

4. Chuyển đồ chơi sang MTP song song (kiểu Gloeckle): thêm đầu ra D lên trên trạng thái ẩn chính, mỗi đầu dự đoán một độ lệch khác nhau. Đo lường mức độ tổn thất trên mỗi độ sâu so với phiên bản tuần tự trên cùng một tín hiệu tổng hợp. Phiên bản tuần tự sẽ tạo ra loss độ sâu k thấp hơn cho k > 1 vì nó điều kiện dựa trên các dự đoán trung gian.

5. Sử dụng mô-đun MTP được huấn luyện như một bản nháp kiểu EAGLE: gọi mô-đun k để đề xuất `t_{i+k}` tại inference. Đo lường tỷ lệ chấp nhận của các tokens nháp này so với dự đoán của model chính trên một trình tự bị giữ lại. Nếu bạn đạt 50%+ trên đồ chơi, bạn đã tái tạo thuộc tính MTP-as-draft thực nghiệm.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Mô-đun MTP | "Khối loss bổ sung" | Một khối transformer nhỏ cộng với phép chiếu dự đoán vị trí token `k` trước model chính |
| Độ sâu dự đoán | "Bù đắp nào" | Số nguyên `k` sao cho mô-đun `k` dự đoán `t_{i+k}` từ tiền tố đến vị trí `i` |
| MTP song song | "Phong cách Gloeckle" | D đầu độc lập trên cùng một trạng thái ẩn xương sống, không có chuỗi có điều kiện |
| MTP tuần tự | "Phong cách DeepSeek-V3" | Mỗi mô-đun điều kiện về trạng thái ẩn của độ sâu trước đó cộng với embedding của token tiếp theo; bảo tồn chuỗi nhân quả |
| Đầu ra được chia sẻ | "Tái sử dụng đầu chính" | Các mô-đun MTP gọi đầu LM của model chính, không phải là phép chiếu đầu ra riêng biệt |
| Chia sẻ embedding | "Tái sử dụng bàn chính" | Cùng một từ vựng embedding bảng được sử dụng ở khắp mọi nơi; không có parameters trùng lặp |
| Ma trận chiếu M_k | "Kết hợp ẩn + token tiếp theo" | Một lớp tuyến tính `h x 2h` gấp trạng thái ẩn trước đó và token embedding mục tiêu vào đầu vào của độ sâu tiếp theo |
| loss L_MTP chung | "Tổn thất thêm trung bình" | Giá trị trung bình cộng của tổn thất entropy chéo trên mỗi độ sâu, được chia tỷ lệ theo `lambda` |
| Tỷ lệ chấp nhận ở độ sâu 1 | "Tần suất dự thảo MTP đúng" | Tỷ lệ dự đoán top 1 của mô-đun D = 1 MTP bằng với dự đoán top 1 của model chính; 80% + trên DeepSeek-V3 |
| Trọng lượng Lambda | "Tầm quan trọng loss hơn" | Hệ số tỷ lệ trên mỗi độ sâu; 0,3 khi bắt đầu training, 0,1 sau đó trên DeepSeek-V3 |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — mô tả MTP tuần tự đầy đủ (Mục 2.2), bao gồm các phương trình loss chung và tăng tốc 1,8× ở inference
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) - đường cơ sở MTP song song mà thiết kế của DeepSeek cải thiện
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) - Tổng cộng 685 tỷ (671 tỷ chính + 14 tỷ MTP), ghi chú triển khai
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — giải mã suy đoán framework MTP phù hợp với
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) - Kiến trúc dự thảo năm 2025 của EAGLE, đối tác MTP cạnh tranh với
