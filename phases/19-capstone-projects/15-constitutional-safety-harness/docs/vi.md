# Capstone 15 - Harness An toàn Hiến pháp + Phạm vi Đội đỏ

> Bộ phân loại hiến pháp của Anthropic, Llama Guard 4 của Meta, ShieldGemma-2 của Google, Nemotron 3 Content Safety của NVIDIA và X-Guard cho phạm vi đa ngôn ngữ đã xác định stack phân loại an toàn năm 2026. garak, PyRIT NVIDIA Aegis và promptfoo đã trở thành các công cụ đánh giá đối thủ tiêu chuẩn. NeMo Guardrails v0.12 gắn chúng thành một production pipeline. Capstone này kết nối tất cả lại với nhau: một harness an toàn nhiều lớp xung quanh một ứng dụng mục tiêu, một đội đỏ tự trị agent chạy 6+ họ tấn công và một cuộc tự phê bình hiến pháp tạo ra một delta vô hại có thể đo lường được.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (safety pipeline, red team), YAML (policy configs)
**Kiến thức tiên quyết:** Giai đoạn 10 (LLMs từ đầu), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 13 (công cụ), Giai đoạn 14 (agents), Giai đoạn 18 (đạo đức, an toàn, alignment)
**Các giai đoạn được thực hiện:** P10 · P11 · P13 · P14 · Trang 18
**Thời lượng:** 25 giờ

## Vấn đề

Ranh giới của sự an toàn LLM vào năm 2026 không phải là liệu các bộ phân loại có hoạt động hay không (đại khái) mà là làm thế nào để soạn chúng một cách chính xác xung quanh một ứng dụng production mà không từ chối quá mức hoặc để lại những lỗ hổng rõ ràng. Llama Guard 4 xử lý các vi phạm policy tiếng Anh. X-Guard (132 ngôn ngữ) xử lý bẻ khóa đa ngôn ngữ. ShieldGemma-2 bắt chèn prompt dựa trên hình ảnh. NVIDIA Nemotron 3 Content Safety bao gồm các danh mục doanh nghiệp. Bộ phân loại hiến pháp của Anthropic là một cách tiếp cận riêng biệt được sử dụng trong training hơn là phục vụ.

Sự tiến hóa của cuộc tấn công cũng rất quan trọng. PAIR và TAP tự động khám phá bẻ khóa. GCG chạy các cuộc tấn công hậu tố dựa trên gradient. Các cuộc tấn công nhiều lượt và chuyển đổi mã khai thác bộ nhớ agent. Bất kỳ LLM nào được triển khai đều cần một phạm vi đội đỏ - garak và PyRIT là những trình điều khiển chính thức - cùng với các biện pháp giảm thiểu được ghi lại và các phát hiện được chấm điểm CVSS.

Bạn sẽ củng cố một ứng dụng mục tiêu (model được điều chỉnh lệnh 8B hoặc một trong những chatbot RAG từ các capstone khác), chạy 6+ họ tấn công chống lại nó và tạo ra phép đo before/after vô hại.

## Khái niệm

pipeline an toàn là năm lớp. **Đầu vào vệ sinh**: loại bỏ các ký tự có chiều rộng bằng không, giải mã base64/rot13, chuẩn hóa Unicode. **Lớp Policy**: Đường ray NeMo Guardrails v0.12 (ngoài miền, độc tính, trích xuất PII). **Cổng phân loại**: Llama Guard 4 trên đầu vào, X-Guard trên không phải tiếng Anh, ShieldGemma-2 trên đầu vào hình ảnh. **Model**: mục tiêu LLM. **Bộ lọc đầu ra**: Llama Guard 4 trên đầu ra, quét Presidio PII, thực thi trích dẫn nếu có. **Bậc HITL**: đầu ra được gắn cờ rủi ro cao sẽ chuyển đến hàng đợi Slack.

Phạm vi đội đỏ chạy trên một bộ lập lịch. PAIR và TAP tự động khám phá bẻ khóa. GCG chạy các cuộc tấn công hậu tố dựa trên gradient. Các cuộc tấn công mã hóa ASCII / base64 / rot13. Tấn công nhiều lượt (nhận nuôi nhân vật, khai thác trí nhớ). Các cuộc tấn công chuyển đổi mã (kết hợp tiếng Anh với tiếng Swahili hoặc tiếng Thái). Mỗi lần chạy tạo ra một tệp phát hiện có cấu trúc với thời gian chấm điểm và tiết lộ CVSS.

Cuộc chạy tự phê bình hiến pháp là một sự can thiệp training lần. Lấy 1k prompts cố gắng gây hại, yêu cầu model soạn thảo phản hồi, phê bình nó dựa trên hiến pháp thành văn (quy tắc không gây hại) và huấn luyện lại về vòng lặp phê bình. Đo lường delta before/after vô hại trên một đánh giá được giữ lại.

## Kiến trúc

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Stack

- Bộ phân loại an toàn: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Guardrail framework: NeMo Guardrails v0.12 + OPA
- Trình điều khiển đội đỏ: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Jailbreak agents: PAIR (Chao và cộng sự, 2023), Tree-of-Attacks (TAP), hậu tố GCG
- training hiến pháp: Vòng lặp tự phê bình kiểu Anthropic + SFT về phê bình
- Tẩy tế bào chết PII: Presidio
- Mục tiêu: một model được điều chỉnh lệnh 8B hoặc một trong những chatbot RAG của capstone khác

## Tự xây dựng

1. **Thiết lập mục tiêu.** Dựng một model được điều chỉnh lệnh 8B trên vLLM (hoặc sử dụng lại chatbot RAG từ một capstone khác). Đây là ứng dụng đang được thử nghiệm.

2. **An toàn pipeline bọc. **Nối dây pipeline năm lớp xung quanh mục tiêu. Xác minh rằng mỗi lớp có thể quan sát riêng lẻ (span mỗi lớp trong Langfuse).

3. **Phạm vi của bộ phân loại.** Load Llama Guard 4, X-Guard (đa ngôn ngữ), ShieldGemma-2 (hình ảnh). Chạy từng tập hợp trên một tập hợp nhỏ được dán nhãn để thiết lập đường cơ sở.

4. **Lập lịch đội đỏ.** Lên lịch garak, PyRIT, một agent PAIR, một agent TAP, một người chạy GCG, một kẻ tấn công nhiều lượt và một kẻ tấn công chuyển mã. Mỗi hàng chạy trên một hàng đợi riêng biệt.

5. **Bộ tấn công.** Sáu họ tấn công: (1) bẻ khóa tự động PAIR, (2) cây tấn công TAP, (3) hậu tố gradient GCG, (4) mã hóa ASCII / base64 / rot13, (5) tính cách nhiều lượt, (6) chuyển đổi mã đa ngôn ngữ. Báo cáo tỷ lệ thành công cho mỗi gia đình.

6. **Tự phê bình hiến pháp.** Sắp xếp 1k prompts cố gắng gây hại. Đối với mỗi mục tiêu, mục tiêu soạn thảo một phản hồi. Một nhà phê bình LLM điểm chống lại hiến pháp thành văn ("không gây hại", "trích dẫn bằng chứng", "từ chối các yêu cầu bất hợp pháp"). Prompts nơi các đối tượng phê bình được viết lại; Mục tiêu tinh chỉnh các cặp được cải thiện phê bình. Đo lường before/after tính vô hại trên một đánh giá được giữ lại.

7. **Đo lường từ chối quá mức.** Theo dõi tỷ lệ dương tính giả trên bộ prompt lành tính (ví dụ: XSTest). Mục tiêu phải hữu ích cho những câu hỏi lành tính.

8. **Chấm điểm CVSS.** Đối với mỗi lần bẻ khóa thành công, hãy chấm điểm trên CVSS 4.0 (vector tấn công, độ phức tạp, tác động). Xây dựng lịch trình tiết lộ và kế hoạch giảm thiểu.

9. **Tự động hóa phạm vi.** Mọi thứ ở trên đều chạy trên một cron; phát hiện viết vào hàng đợi; hồi quy từ chối quá mức cảnh báo cháy Slack.

## Ứng dụng

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Sản phẩm bàn giao

`outputs/skill-safety-harness.md` là sản phẩm được giao. Một pipeline an toàn phân lớp cấp production cộng với phạm vi đội đỏ có thể tái tạo với các delta before/after vô hại.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Phạm vi bao phủ bề mặt tấn công | 6+ gia đình tấn công được thực hiện, 2+ ngôn ngữ |
| 20 | Đánh đổi dương tính đúng / dương tính giả | Tỷ lệ chặn tấn công so với XSTtỷ lệ vượt qua lành tính nhất |
| 20 | Tự phê bình delta | Before/after vô hại khi đánh giá |
| 20 | Tài liệu và tiết lộ | Các phát hiện được chấm điểm CVSS với dòng thời gian |
| 15 | Tự động hóa và độ lặp lại | Mọi thứ chạy trên cron với cảnh báo |
| **100** |||

## Bài tập

1. Chạy plugin của garak để tiêm prompt trên chatbot RAG và so sánh tỷ lệ tấn công thành công có và không có lớp bộ lọc đầu ra.

2. Thêm họ tấn công thứ bảy: chèn prompt gián tiếp thông qua các tài liệu được truy xuất. Đo lường khả năng phòng thủ bổ sung cần thiết.

3. Thực hiện chế độ "từ chối có trợ giúp": khi guardrail chặn, mục tiêu sẽ đưa ra câu trả lời liên quan an toàn hơn thay vì từ chối thẳng thừng. Đo XSTest delta.

4. Khoảng cách phạm vi đa ngôn ngữ: tìm ngôn ngữ mà X-Guard hoạt động kém hiệu quả. Đề xuất một fine-tune dataset nhắm mục tiêu vào nó.

5. Chạy bản tự phê bình hiến pháp trên model 30B và đo lường xem delta có quy mô hay không.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| An toàn nhiều lớp | "Phòng thủ theo chiều sâu" | Nhiều guardrails ở đầu vào, cổng, đầu ra, HITL |
| Bảo vệ Llama 4 | "Bộ phân loại an toàn của Meta" | Bộ phân loại nội dung input/output tham chiếu năm 2026 |
| CẶP | "Jailbreak agent" | Bài báo (Chao và cộng sự) về khám phá bẻ khóa do LLM điều khiển |
| CHẠM VÀO | "Cây tấn công" | Biến thể tìm kiếm cây của PAIR |
| GCG | "Tọa độ tham lam gradient" | Tấn công hậu tố đối kháng dựa trên Gradient |
| Tự phê bình hiến pháp | "training kiểu Anthropic" | Bản nháp mục tiêu -> điểm phê bình -> viết lại -> huấn luyện lại |
| XSTest | "Bộ đầu dò lành tính" | Benchmark cho hồi quy từ chối quá mức |
| CVSS 4.0 | "Điểm mức độ nghiêm trọng" | Chấm điểm lỗ hổng bảo mật tiêu chuẩn cho các phát hiện an toàn |

## Đọc thêm

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — tham chiếu training thời gian
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — bộ phân loại input/output năm 2026
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — hình ảnh + an toàn đa phương thức
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — Tài liệu tham khảo doanh nghiệp
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — An toàn đa ngôn ngữ 132 ngôn ngữ
- [garak](https://github.com/NVIDIA/garak) — NVIDIA bộ công cụ đội đỏ
- [PyRIT](https://github.com/Azure/PyRIT) — framework đội đỏ của Microsoft
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) - framework đường sắt
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — bẻ khóa agent giấy
