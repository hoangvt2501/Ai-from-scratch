# Đánh giá ngữ cảnh dài - NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro quảng cáo 10 triệu tokens ngữ cảnh. Ở tokens 1 triệu, MRCR 8 kim giảm xuống còn 26,3%. Quảng cáo ≠ sử dụng được. Đánh giá ngữ cảnh dài cho bạn biết năng lực thực tế của model bạn đang shipping.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 13 (Trả lời câu hỏi), Giai đoạn 5 · 23 (Chiến lược phân đoạn)
**Thời lượng:** ~60 phút

## Vấn đề

Bạn có một hợp đồng dài 200 trang. model tuyên bố bối cảnh 1 triệu token. Bạn dán hợp đồng vào và hỏi: "Điều khoản chấm dứt là gì?" Câu trả lời model - nhưng câu trả lời từ trang bìa vì điều khoản chấm dứt nằm ở độ sâu 120k tokens, vượt qua nơi model thực sự tham dự.

Đây là khoảng cách dung lượng ngữ cảnh năm 2026. Bảng thông số kỹ thuật cho biết 1M hoặc 10M. Thực tế nói rằng 60-70% trong số đó có thể sử dụng được và "có thể sử dụng" phụ thuộc vào nhiệm vụ.

- **Truy xuất (kim đơn trong đống cỏ khô):** gần như hoàn hảo lên đến mức tối đa được quảng cáo trên models biên giới.
- **Multi-hop / tổng hợp: **giảm mạnh vượt qua ~128k trên hầu hết các models.
- **Lý luận về các sự kiện phân tán: **nhiệm vụ đầu tiên thất bại.

Đánh giá ngữ cảnh dài đo lường các trục này. Bài học này đặt tên cho các benchmarks, những gì mỗi loại thực sự đo lường và cách xây dựng một thử nghiệm kim tùy chỉnh cho miền của bạn.

## Khái niệm

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Đặt một sự thật ("từ ma thuật là dứa") ở độ sâu được kiểm soát trong một ngữ cảnh dài. Yêu cầu model lấy nó. Độ sâu × chiều dài quét. Bối cảnh dài ban đầu benchmark. Frontier bây giờ models bão hòa điều này; đó là một đường cơ sở cần thiết nhưng không đủ.

**THƯỚC KẺ (Nvidia, 2024).** 13 loại tác vụ trên 4 danh mục: truy xuất (đơn / đa phím / đa giá trị), theo dõi nhiều bước (theo dõi biến), tổng hợp (tần suất từ phổ biến), QA. Độ dài ngữ cảnh có thể định cấu hình (4k đến 128k+). Tiết lộ models bão hòa NIAH nhưng thất bại trong nhiều bước nhảy. Trong bản phát hành năm 2024, chỉ một nửa trong số 17 models tuyên bố bối cảnh 32k+ duy trì chất lượng ở mức 32k.

**LongBench v2 (2024).** 503 câu hỏi trắc nghiệm, ngữ cảnh từ 8k-2 triệu, sáu danh mục nhiệm vụ: QA một tài liệu, QA nhiều tài liệu, học trong ngữ cảnh dài, đối thoại dài, repo mã, dữ liệu có cấu trúc dài. production benchmark cho hành vi bối cảnh dài trong thế giới thực.

**MRCR (Độ phân giải đồng tham chiếu nhiều vòng).** Đồng tham chiếu nhiều lượt trên quy mô lớn. Các biến thể 8 kim, 24 kim, 100 kim. Tiết lộ bao nhiêu sự thật mà một model có thể tung hứng trước khi attention xuống cấp.

**NoLiMa.** "Kim không từ vựng." Kim và truy vấn không có sự trùng lặp theo nghĩa đen; Truy xuất đòi hỏi một bước suy luận ngữ nghĩa. Khó hơn NIAH.

**HELMET.** Nối nhiều tài liệu, đặt câu hỏi từ bất kỳ tài liệu nào. Kiểm tra attention chọn lọc.

**BABILong.** Nhúng chuỗi lý luận bAbI bên trong đống cỏ khô không liên quan. Kiểm tra lý luận trong đống cỏ khô, không chỉ truy xuất.

### Những gì thực sự cần báo cáo

- **Được quảng cáo context window.** Số bảng thông số kỹ thuật.
- **Thời gian truy xuất hiệu quả.** NIAH vượt qua ở một số ngưỡng (ví dụ: 90%).
- **Độ dài suy luận hiệu quả.** Vượt qua nhiều bước nhảy hoặc tổng hợp ở ngưỡng đó.
- **Đường cong suy giảm.** Độ dài Accuracy so với ngữ cảnh, được vẽ theo loại tác vụ.

Hai con số cho bảng thông số kỹ thuật của bạn: hiệu quả truy xuất và hiệu quả lý luận. Thông thường, hiệu quả lý luận là 25-50% cửa sổ được quảng cáo.

## Tự xây dựng

### Bước 1: NIAH tùy chỉnh cho miền của bạn

Xem `code/main.py`. Bộ xương:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Quét `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Vẽ bản đồ nhiệt. Đó là thẻ NIAH cho model mục tiêu của bạn.

### Bước 2: biến thể nhiều kim

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Những câu hỏi như "Ba từ ma thuật là gì?" yêu cầu truy xuất cả ba. Thành công bằng một kim không dự đoán được thành công bằng nhiều kim.

### Bước 3: theo dõi biến nhiều bước (kiểu RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Câu trả lời yêu cầu xâu chuỗi ba bài tập. Frontier models ở mức 128k thường giảm xuống còn 50-70% accuracy ở đây.

### Bước 4: LongBench v2 trên stack của bạn

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Báo cáo theo từng danh mục accuracy. Điểm tổng hợp che giấu sự khác biệt lớn ở cấp độ nhiệm vụ.

## Cạm bẫy

- **Đánh giá chỉ có NIAH.** Vượt qua NIAH ở mức 1 triệu tokens không nói gì về multi-hop. Luôn chạy RULER hoặc kiểm tra nhiều bước tùy chỉnh.
- **Độ sâu đồng nhất sampling.** Nhiều triển khai chỉ kiểm tra độ sâu = 0,5. Độ sâu thử nghiệm = 0, 0,25, 0,5, 0,75, 1,0 - hiệu ứng "mất giữa chừng" là có thật.
- **Trùng lặp từ vựng với chất độn.** Nếu kim chia sẻ từ khóa với chất độn, việc truy xuất trở nên tầm thường. Sử dụng kim không chồng chéo kiểu NoLiMa.
- **Bỏ qua độ trễ.** 1M-token prompts mất 30-120 giây để nạp trước. Đo thời gian token đầu tiên cùng với accuracy.
- **Số liệu do nhà cung cấp tự báo cáo.** OpenAI, Google Anthropic tất cả đều công bố điểm số của riêng họ. Luôn chạy lại độc lập trên trường hợp sử dụng của bạn.

## Ứng dụng

stack năm 2026:

| Tình huống | Benchmark |
|-----------|-----------|
| Kiểm tra nhanh sự tỉnh táo | NIAH tùy chỉnh ở 3 độ sâu × 3 chiều dài |
| Lựa chọn Model cho production | THƯỚC KẺ (13 nhiệm vụ) ở độ dài mục tiêu của bạn |
| Chất lượng QA trong thế giới thực | Tập hợp con LongBench v2 single-doc-QA |
| Lý luận nhiều bước nhảy | BABILong hoặc theo dõi biến tùy chỉnh |
| Đàm thoại / đối thoại | MRCR 8 kim ở chiều dài mục tiêu của bạn |
| Hồi quy nâng cấp Model | Cố định harness NIAH + RULER nội bộ, chạy trên mọi model mới |

Quy tắc chung cho production: đừng bao giờ tin tưởng một context window cho đến khi bạn có nhiệm vụ suy luận NIAH + 1 với độ dài dự định của bạn.

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Bài tập

1. **Dễ dàng.** Xây dựng NIAH với 3 độ sâu (0,25, 0,5, 0,75) × 3 độ dài (1k, 4k, 16k). Chạy trên bất kỳ model nào. Tỷ lệ vượt qua biểu đồ dưới dạng bản đồ nhiệt 3×3.
2. **Trung bình.** Thêm biến thể 3 kim. Đo khả năng truy xuất của cả 3 ở mỗi độ dài. So sánh với tỷ lệ vượt qua một kim ở cùng độ dài.
3. **Khó.** Xây dựng một tác vụ theo dõi biến đổi (X1 → X2 → X3, với 3 bước nhảy) được nhúng trong 64k chất độn. Đo accuracy trên 3 models biên giới. Báo cáo độ dài suy luận hiệu quả trên mỗi model.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| NIAH | Kim trong đống cỏ khô | Trồng một thực tế vào chất độn, yêu cầu model lấy nó. |
| THƯỚC KẺ | NIAH trên steroid | 13 loại tác vụ trên truy xuất / đa bước / tổng hợp / QA. |
| Bối cảnh hiệu quả | Năng lực thực sự | Độ dài mà tại đó accuracy vẫn giữ trên ngưỡng. |
| Lạc giữa | Độ sâu bias | Models không quan tâm đến nội dung giữa các đầu vào dài. |
| Nhiều kim | Nhiều sự thật cùng một lúc | Nhiều cây; các bài kiểm tra attention tung hứng, không phải truy xuất một mình. |
| MRCR | Coref nhiều vòng | Đồng tham chiếu 8, 24 hoặc 100 kim; để lộ độ bão hòa attention. |
| NoLiMa | Kim không từ vựng | Kim và truy vấn không chia sẻ tokens nghĩa đen; đòi hỏi lý luận. |

## Đọc thêm

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) - repo NIAH ban đầu.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) - benchmark đa nhiệm.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — đánh giá bối cảnh dài trong thế giới thực.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) - kim cứng hơn.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) - lý luận trong đống cỏ khô.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - giấy bias độ sâu.
