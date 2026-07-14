# Đánh giá - FID, Điểm CLIP, Sở thích của con người

> Mọi bảng xếp hạng model tổng quát đều trích dẫn FID, điểm CLIP và tỷ lệ thắng từ đấu trường ưa thích của con người. Mỗi con số có một chế độ thất bại mà một nhà nghiên cứu quyết tâm có thể chơi game. Nếu bạn không biết các chế độ lỗi, bạn không thể nhận ra sự cải thiện thực sự từ khi chơi game.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 01 (Phân loại), Giai đoạn 2 · 04 (Chỉ số đánh giá)
**Thời lượng:** ~45 phút

## Vấn đề

Một model tổng quát được đánh giá dựa trên * chất lượng mẫu * và * tuân thủ điều kiện *. Cả hai đều không có biện pháp dạng đóng. model của bạn phải hiển thị 10.000 hình ảnh; một cái gì đó phải gán cho họ những con số; Bạn phải tin tưởng vào các con số của model họ, giữa các độ phân giải, trên các kiến trúc. Ba chỉ số vẫn tồn tại trong giai đoạn 2014-2026:

- **FID (Fréchet Inception Distance).** Khoảng cách giữa hai phân phối - thực và được tạo ra - trong không gian feature của mạng Inception. Thấp hơn là tốt hơn.
- **Điểm CLIP.** Sự tương đồng cosin giữa embedding hình ảnh CLIP của hình ảnh được tạo và embedding văn bản CLIP của prompt. Cao hơn là tốt hơn. Các biện pháp prompt tuân thủ.
- **Sở thích của con người.** Đấu hai models đối đầu trên cùng một prompt, để con người (hoặc GPT-4-class model) chọn cái tốt hơn, tổng hợp thành điểm Elo.

Bạn cũng sẽ thấy: IS (điểm khởi đầu, phần lớn đã nghỉ hưu), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Mỗi người sửa chữa cho một lỗi của lỗi trước đó.

## Khái niệm

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — chất lượng mẫu

Heusel và cộng sự (2017). Các bước:

1. Trích xuất features Inception-v3 (2048-D) cho N hình ảnh thực và N được tạo.
2. Phù hợp với một Gaussian cho mỗi nhóm: tính toán trung bình `μ_r, μ_g` và hiệp phương sai `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Giải thích: Khoảng cách Fréchet giữa hai Gaussian đa biến trong không gian feature. Thấp hơn = nhiều phân phối tương tự hơn.

Chế độ thất bại:
- **Thiên vị trên N nhỏ **FID bình phương trung bình trên phân phối feature - N nhỏ ước tính thấp hiệp phương sai, cho FID thấp sai. Luôn sử dụng N ≥ 10,000.
- **Phụ thuộc vào Inception.** Inception-v3 được huấn luyện trên ImageNet. Các miền xa ImageNet (khuôn mặt, nghệ thuật, hình ảnh văn bản) tạo ra FID vô nghĩa. Sử dụng trình trích xuất feature dành riêng cho miền.
- **Chơi game.** Overfitting với Inception prior cho độ FID thấp mà không cải thiện chất lượng hình ảnh. Đánh bại nó bằng CMMD (bên dưới).

### Điểm CLIP - tuân thủ prompt

Radford và cộng sự (2021). Đối với hình ảnh + prompt được tạo:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Trung bình trên 30k hình ảnh được tạo → một vô hướng có thể so sánh giữa models.

Chế độ thất bại:
- **Điểm mù của riêng CLIP.** CLIP có lý luận bố cục yếu ("một khối lập phương màu đỏ trên một quả cầu màu xanh" thường thất bại). Models có thể xếp hạng tốt trên điểm CLIP mà không thực sự tuân theo các prompts phức tạp.
- **Short prompt bias.** Short prompts có nhiều hình ảnh CLIP phù hợp hơn trong tự nhiên. Lâu hơn prompts có điểm CLIP thấp hơn về mặt máy móc.
- **Prompt chơi game.** Bao gồm "chất lượng cao, 4k, kiệt tác" trong prompt làm tăng điểm CLIP mà không cải thiện liên kết hình ảnh-văn bản.

CMMD (Jayasumana và cộng sự, 2024) khắc phục một số điều sau: sử dụng features CLIP thay vì Inception, chênh lệch trung bình tối đa thay vì Fréchet. Phát hiện tốt hơn sự khác biệt về chất lượng tinh tế.

### Sở thích của con người - ground truth

Chọn một nhóm prompts. Tạo bằng model A và model B. Hiển thị các cặp cho con người (hoặc một giám khảo LLM mạnh mẽ). Tổng số chiến thắng thành điểm Elo hoặc Bradley-Terry. Benchmarks:

- **PartiPrompts (Google)**: 1.600 prompts đa dạng, 12 danh mục.
- **HPSv2**: 107k chú thích của con người, được sử dụng rộng rãi dưới dạng proxy tự động.
- **ImageReward**: 137 nghìn cặp tùy chọn hình ảnh prompt, được MIT cấp phép.
- **PickScore**: được huấn luyện về tùy chọn Pick-a-Pic 2.6 triệu.
- **Đấu trường hình ảnh theo phong cách Chatbot-Arena**: https://imagearena.ai/ và những người khác.

Chế độ thất bại:
- **Thẩm phán variance.** Những người không phải là chuyên gia có sở thích khác với các chuyên gia. Sử dụng cả hai.
- **Prompt phân phối.** Anh đào chọn prompts ủng hộ một gia đình. Luôn ghi lại.
- **Hack phần thưởng LLM giám khảo.** GPT-4-judge bị lừa bởi những kết quả đẹp nhưng sai. Tam giác với con người.

## Sử dụng cùng nhau

Báo cáo đánh giá production nên bao gồm:

1. FID trên 10-30k mẫu so với phân phối thực tế (chất lượng mẫu).
2. Điểm CLIP / CMMD trên cùng một samples so với prompts của chúng (tuân thủ).
3. Tỷ lệ thắng trong đấu trường mù so với model trước đó (ưu tiên tổng thể).
4. Phân tích chế độ lỗi: 50 đầu ra được lấy mẫu ngẫu nhiên, được gắn cờ cho các vấn đề đã biết (giải phẫu bàn tay, hiển thị văn bản, số lượng đối tượng nhất quán).

Bất kỳ số liệu đơn lẻ nào cũng là dối trá. Ba số liệu xác thực + đánh giá định tính là một tuyên bố.

## Tự xây dựng

`code/main.py` triển khai tổng hợp FID, CLIP-score-like và Elo trên "feature vectors" tổng hợp (chúng tôi sử dụng vectors 4-D làm thay thế cho Inception features). Bạn thấy đấy:

- Tính toán FID trên một N nhỏ và trên một N lớn - bias.
- "Điểm CLIP" là sự tương đồng cosin giữa các nhóm feature.
- Quy tắc cập nhật Elo từ luồng tùy chọn tổng hợp.

### Bước 1: FID trong bốn dòng

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Bước 2: Tương tự cosin kiểu CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Bước 3: Tổng hợp Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Cạm bẫy

- **FID ở N = 1000.** Heuristic không đáng tin cậy dưới N = 10k. Các bài báo báo cáo FID N thấp đang chơi game.
- **So sánh FID giữa các độ phân giải.** Thay đổi kích thước 299×299 của Inception thay đổi phân phối feature. Chỉ so sánh ở độ phân giải phù hợp.
- **Báo cáo một hạt giống.** Chạy tối thiểu 3 hạt. Báo cáo std.
- **Lạm phát điểm CLIP thông qua prompts âm.** Một số pipelines tăng CLIP bằng cách quá vừa vặn với prompt. Kiểm tra độ bão hòa thị giác.
- **Elo bias từ prompt chồng chéo.** Nếu cả hai models đều nhìn thấy một benchmark prompt trong training, Elo là vô nghĩa. Sử dụng bộ prompt giữ lại.
- **Sai lệch đám đông trả tiền của con người.** Các chú thích MTURK sung mãn, nghiêng về trẻ hơn / thân thiện với công nghệ. Kết hợp với các chuyên gia art/design được tuyển dụng.

## Ứng dụng

Production giao thức đánh giá vào năm 2026:

| Trụ cột | Tối thiểu | Đề xuất |
|--------|---------|-------------|
| Chất lượng mẫu | FID trên 10k so với thực tế | + CMMD trên 5k + FID trên tập hợp con cho mỗi danh mục |
| Prompt tuân thủ | Điểm CLIP trên 30k | + HPSv2 + ImageReward + Trả lời câu hỏi theo phong cách VQA |
| Sở thích | 200 cặp mù so với đường cơ sở | + 2000 cặp người + LLM giám khảo + Chatbot Arena |
| Phân tích lỗi | 50 gắn cờ bằng tay | 500 máy phân loại an toàn tự động + gắn cờ thủ công |

Tất cả bốn trụ cột trong một báo cáo = yêu cầu. Bất kỳ một mình nào = tiếp thị.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-eval-report.md`. Skill lấy một model checkpoint + đường cơ sở mới và đưa ra một kế hoạch đánh giá đầy đủ: kích thước mẫu, số liệu, đầu dò chế độ lỗi, tiêu chí đăng xuất.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. So sánh FID ở N = 100 so với N = 1000 trên cùng một phân phối tổng hợp. Báo cáo bias độ lớn.
2. **Trung bình.** Triển khai CMMD từ features kiểu CLIP tổng hợp (xem Jayasumana et al., 2024 để biết công thức). So sánh độ nhạy với sự khác biệt về chất lượng so với FID.
3. **Khó khăn.** Sao chép thiết lập HPSv2: lấy 1000 cặp prompt hình ảnh từ một tập hợp con của Pick-a-Pic, fine-tune một bộ ghi điểm dựa trên CLIP nhỏ theo sở thích và đo lường sự phù hợp của nó với một tập hợp bị giữ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| FID | "Khoảng cách khởi đầu Fréchet" | Khoảng cách Fréchet của Gaussian phù hợp với thực so với thế hệ Inception features. |
| Điểm CLIP | "Sự giống nhau giữa văn bản và hình ảnh" | Sự tương đồng cosin giữa hình ảnh CLIP và embeddings văn bản. |
| CMMD | "Thay thế FID" | CLIP-feature MMD; ít thiên vị hơn, không có giả định Gaussian. |
| LÀ | "Điểm khởi động" | Kinh nghiệm KL (p (y | x) || p(y)); tương quan kém với models hiện đại, đã nghỉ hưu. |
| HPSv2 / Phần thưởng hình ảnh / Điểm chọn | "Proxy ưu tiên đã học" | Các models nhỏ được huấn luyện theo sở thích của con người; được sử dụng làm thẩm phán tự động. |
| Elo | "Đánh giá cờ vua" | Bradley-Terry tổng hợp chiến thắng theo cặp. |
| Lời nhắc PartiPrompts | "Bộ benchmark prompt" | 1.600 prompts do Google tuyển chọn trên 12 danh mục. |
| FD-DINO | "Tự thay thế" | FD sử dụng DINOv2 features; tốt hơn cho các miền ngoài ImageNet. |

## Production lưu ý: đánh giá cũng là một khối lượng công việc inference

Chạy FID trên 10k mẫu có nghĩa là tạo ra 10k hình ảnh. Đối với đế SDXL 50 bước ở 1024² trên một L4 duy nhất, đó là ~11 giờ inference yêu cầu đơn. Ngân sách đánh giá là có thật và việc đóng khung chính xác là kịch bản inference ngoại tuyến (tối đa hóa thông lượng, bỏ qua TTFT):

- **Batch khó, quên độ trễ.** Đánh giá ngoại tuyến = hàng loạt tĩnh ở kích thước lớn nhất phù hợp với bộ nhớ. `pipe(...).images` với `num_images_per_prompt=8` trên H100 80GB chạy xung nhịp treo tường nhanh hơn 4-6× so với một yêu cầu.
- **Lưu trữ features thực.** Trích xuất feature Inception (FID) hoặc CLIP (CLIP-score, CMMD) trên tập tham chiếu thực được chạy *một lần*, được lưu trữ dưới dạng `.npz`. Không tính toán lại mỗi lần đánh giá.

Đối với cổng CI / hồi quy: chạy điểm FID + CLIP trên tập con 500 mẫu mỗi PR (~30 phút); chạy đầy đủ 10k FID + HPSv2 + Elo hàng đêm.

## Đọc thêm

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — Giấy FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) - CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) - HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) - Phần thưởng hình ảnh.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) - PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) - khảo sát chế độ thất bại.
