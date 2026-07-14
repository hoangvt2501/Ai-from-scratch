# Capstone 08 — Production RAG Chatbot cho một ngành dọc được quản lý

> Harvey, Glean, Mendable và LlamaCloud đều chạy cùng một hình dạng production vào năm 2026. Nhập bằng docling hoặc Unstructured và ColPali cho hình ảnh. Tìm kiếm kết hợp. Xếp hạng lại với bge-reranker-v2-gemma. Tổng hợp với Claude Sonnet 4.7 bằng cách sử dụng bộ nhớ đệm prompt với tỷ lệ trúng 60-80%. Bảo vệ với Llama Guard 4 và NeMo Guardrails. Xem với Langfuse và Phoenix. Chấm điểm với RAGAS trên một bộ vàng gồm 200 câu hỏi. Xây dựng một trong một lĩnh vực được quy định (pháp lý, lâm sàng, bảo hiểm) và capstone sẽ vượt qua bộ vàng, đội đỏ và bảng điều khiển trôi dạt.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (pipeline + API), TypeScript (chat UI)
**Kiến thức tiên quyết:** Giai đoạn 5 (NLP), Giai đoạn 7 (transformers), Giai đoạn 11 (LLM kỹ thuật), Giai đoạn 12 (đa phương thức), Giai đoạn 17 (cơ sở hạ tầng), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P5 · P7 · P11 · P12 · P17 · Trang 18
**Thời lượng:** 30 giờ

## Vấn đề

RAG miền được quy định (hợp đồng pháp lý, giao thức thử nghiệm lâm sàng, policies bảo hiểm) là hình dạng shipped production nhất của năm 2026 vì ROI là rõ ràng và cổ phần là cụ thể. Harvey (Allen & Overy) đã xây dựng nó để hợp pháp. Có thể sửa chữa ships hương vị tài liệu dành cho nhà phát triển. Glean bao gồm tìm kiếm doanh nghiệp. Mô hình là: nhập độ trung thực cao, truy xuất lai với xếp hạng lại, tổng hợp với thực thi trích dẫn và bộ nhớ đệm prompt, bảo vệ với nhiều lớp an toàn và theo dõi trôi liên tục.

Các phần cứng không phải là model. Đó là tuân thủ nhận thức được khu vực pháp lý (HIPAA, GDPR, SOC2), khả năng kiểm tra cấp trích dẫn, kiểm soát chi phí (prompt bộ nhớ đệm mua chiết khấu 60-90% khi tỷ lệ truy cập cao), phát hiện ảo giác thông qua độ trung thực của RAGAS và phát hiện trôi dạt khi tài liệu nguồn được cập nhật mà chỉ mục không bắt kịp. Capstone này yêu cầu bạn ship tất cả trên một bộ vàng gồm 200 câu hỏi với một bộ đội màu đỏ bên cạnh.

## Khái niệm

pipeline có hai mặt. **Nhập**: lập tài liệu hoặc phân tích cú pháp phi cấu trúc các tài liệu có cấu trúc; ColPali xử lý những cái giàu hình ảnh; Các phần nhận được bản tóm tắt, thẻ và nhãn truy cập dựa trên vai trò. Vectors vào pgvector + pgvectorscale (dưới 50M vectors) hoặc Qdrant Cloud; BM25 thưa thớt chạy bên cạnh. **Hội thoại**: LangGraph xử lý bộ nhớ và nhiều lượt; Mỗi truy vấn chạy truy xuất kết hợp, xếp hạng lại với bge-reranker-v2-gemma-2b, tổng hợp với Claude Sonnet 4.7 (prompt-cached), chuyển đầu ra qua Llama Guard 4 và NeMo Guardrails và phát ra phản hồi neo trích dẫn.

stack đánh giá có bốn lớp. **Bộ vàng** (200 Q/A được dán nhãn có trích dẫn) cho tính chính xác. **Đội đỏ** (bẻ khóa, nỗ lực trích xuất PII, câu hỏi ngoài miền) để đảm bảo an toàn. **RAGAS** cho độ trung thực / mức độ liên quan của câu trả lời / ngữ cảnh precision tự động mỗi lượt. **Bảng điều khiển trôi dạt** (Arize Phoenix) xem chất lượng truy xuất và điểm ảo giác hàng tuần.

Prompt bộ nhớ đệm là đòn bẩy chi phí. Claude 4.5+ và GPT-5+ hỗ trợ hệ thống bộ nhớ đệm prompts + ngữ cảnh được truy xuất. Với tỷ lệ truy cập 60-80%, chi phí cho mỗi truy vấn giảm 3-5 lần. pipeline phải được thiết kế cho các tiền tố ổn định (system prompt + ngữ cảnh được xếp hạng lại trước) để đạt được tỷ lệ truy cập bộ nhớ đệm cao.

## Kiến trúc

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack

- Nhập: Unstructured.io hoặc docling cho các tài liệu có cấu trúc; ColPali cho các tệp PDF phong phú về hình ảnh
- Vector DB: pgvector + pgvectorscale dưới 50M vectors; Qdrant Cloud khác
- Thưa thớt: Tantivy BM25 với trọng lượng trường
- Orchestration: Quy trình làm việc LlamaIndex (nhập) + LangGraph (hội thoại)
- Xếp hạng lại: bge-reranker-v2-gemma-2b tự lưu trữ hoặc Voyage rerank-2 được lưu trữ
- LLM: Claude Sonnet 4.7 với bộ nhớ đệm prompt; dự phòng Llama 3.3 70B tự lưu trữ
- Eval: RAGAS 0.2 trực tuyến, DeepEval cho các bộ ảo giác và bẻ khóa
- Observability: Langfuse tự lưu trữ với hàng đợi chú thích; Arize Phoenix để trôi dạt
- Guardrails: Bộ phân loại input/output Llama Guard 4, NeMo Guardrails v0.12 policy, Presidio PII scrub
- Tuân thủ: nhãn truy cập dựa trên vai trò trên các khối; thẻ khu vực tài phán cho GDPR/HIPAA

```figure
canary-rollout
```

## Tự xây dựng

1. **Nhập.** Phân tích cú pháp kho dữ liệu của bạn (1000-10000 tài liệu cho một bản dựng nghiêm túc) với Unstructured hoặc docling. Đối với các trang được quét / nhiều hình ảnh, hãy định tuyến qua ColPali. Tạo các đoạn với tóm tắt, nhãn vai trò, thẻ thẩm quyền.

2. **Index.** embeddings dày đặc (Voyage-3 hoặc Nomic-embed-v2) thành pgvector + pgvectorscale. Chỉ số bên BM25 thông qua Tantivy. Bộ lọc vai trò và khu vực tài phán là payload.

3. **Truy xuất kết hợp.** Lọc theo vai trò + khu vực pháp lý trước; sau đó song song dày đặc + BM25; merge với sự hợp nhất cấp bậc đối ứng; top-20 để xếp hạng lại; TOP-5 để tổng hợp.

4. **Tổng hợp với bộ nhớ đệm prompt.** System prompt + policies tĩnh trong tiêu đề bộ nhớ đệm; xếp hạng lại ngữ cảnh dưới dạng phần mở rộng bộ nhớ đệm; câu hỏi của người dùng dưới dạng hậu tố không được lưu trong bộ nhớ đệm. Mục tiêu 60-80% tỷ lệ trúng bộ nhớ đệm ở trạng thái ổn định.

5. **Guardrails.** Llama Guard 4 trên đầu vào; Đường ray Guardrails NeMo chặn các câu hỏi ngoài tên miền hoặc các chủ đề bị cấm policy; Presidio loại bỏ PII ngẫu nhiên trong đầu ra; thực thi trích dẫn sau bộ lọc.

6. **Bộ vàng.** 200 cặp Q/A được dán nhãn bởi một chuyên gia lĩnh vực với (câu trả lời, trích dẫn). Chấm điểm agent về khớp trích dẫn chính xác, đúng câu trả lời, trung thực (RAGAS).

7. **Đội đỏ.** 50 prompts đối thủ: bẻ khóa (PAIR, TAP), nỗ lực đánh cắp PII, rò rỉ ngoài miền, xuyên khu vực. Ghi điểm với mức độ pass/fail và mức độ nghiêm trọng.

8. **Bảng điều khiển trôi dạt.** Arize Phoenix theo dõi chất lượng truy xuất (nDCG, độ trung thực của trích dẫn) hàng tuần. Cảnh báo khi giảm 5%.

9. **Báo cáo chi phí.** Langfuse: Tỷ lệ truy cập bộ nhớ đệm prompt, tokens cho mỗi truy vấn, phân tích $/truy vấn theo giai đoạn.

## Ứng dụng

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Sản phẩm bàn giao

`outputs/skill-production-rag.md` mô tả sản phẩm. Một chatbot miền được quy định được triển khai với các nhãn tuân thủ, được chuyển qua bảng đánh giá, được quan sát bằng tính năng giám sát trôi dạt trực tiếp.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Độ trung thực của RAGAS + mức độ liên quan của câu trả lời | Điểm số trực tuyến trên bộ vàng (200 Q/A) |
| 20 | Tính đúng đắn của trích dẫn | Phần câu trả lời có neo nguồn có thể xác minh |
| 20 | Bảo hiểm Guardrail | Tỷ lệ vượt qua Llama Guard 4 + kết quả bộ bẻ khóa |
| 20 | Kỹ thuật chi phí / độ trễ | Tỷ lệ truy cập bộ nhớ đệm Prompt, độ trễ p95, $/truy vấn |
| 15 | Bảng điều khiển giám sát trôi dạt | Bảng điều khiển trực tiếp của Phoenix với xu hướng chất lượng truy xuất hàng tuần |
| **100** |||

## Bài tập

1. Xây dựng phần kho dữ liệu thứ hai theo một khu vực pháp lý khác (ví dụ: HIPAA cùng với GDPR). Thể hiện vai trò + khu vực tài phán lọc ngăn chặn rò rỉ chéo trên một cuộc điều tra liên khu vực tài phán gồm 20 câu hỏi.

2. Đo lường tỷ lệ truy cập bộ nhớ đệm prompt trong một tuần lưu lượng truy cập production. Xác định truy vấn nào phá vỡ tiền tố bộ nhớ đệm. Tái cấu trúc.

3. Thêm bộ nhớ nhiều lượt với bộ đệm tóm tắt 10k token. Đo lường xem sự trung thành có giảm khi cuộc trò chuyện phát triển hay không.

4. Hoán đổi Claude Sonnet 4.7 lấy Llama 3.3 70B tự lưu trữ. Đo lường $/truy vấn và delta trung thực.

5. Thêm chế độ "không chắc chắn": nếu điểm số được xếp hạng lại hàng đầu dưới ngưỡng, agent sẽ cho biết "Tôi không có trích dẫn tự tin" thay vì trả lời. Đo lường giảm độ tin cậy giả.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Prompt bộ nhớ đệm | "Hệ thống bộ nhớ cache + ngữ cảnh" | Claude/OpenAI feature: tiền tố được lưu trong bộ nhớ cache tokens giảm giá 60-90% khi trúng |
| RAGAS | "Người đánh giá RAG" | Tự động chấm điểm về độ trung thực, mức độ liên quan của câu trả lời, ngữ cảnh precision |
| Bộ vàng | "Đánh giá được dán nhãn" | 200+ Q/A được gắn nhãn chuyên gia với các trích dẫn; ground truth |
| Thẻ thẩm quyền | "Nhãn tuân thủ" | GDPR/HIPAA/SOC2 phạm vi gắn vào các khối; được thực thi bởi bộ lọc truy xuất |
| Trích dẫn trung thành | "Tỷ lệ trả lời có cơ sở" | Phần lớn các thông báo xác nhận quyền sở hữu được hỗ trợ bởi spans nguồn có thể truy xuất |
| Trôi dạt | "Phân rã chất lượng truy xuất" | Thay đổi hàng tuần về nDCG hoặc điểm trích dẫn; Ngưỡng cảnh báo 5% |
| Đội đỏ | "Đánh giá đối nghịch" | Bẻ khóa trước khi phát hành, trích xuất PII, thăm dò ngoài miền |

## Đọc thêm

- [Harvey AI](https://www.harvey.ai) — tham khảo production stack pháp lý
- [Glean enterprise search](https://www.glean.com) — RAG tham khảo ở quy mô doanh nghiệp
- [Mendable documentation](https://mendable.ai) — tài liệu tham khảo RAG dành cho nhà phát triển
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — nhập được quản lý
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — tham chiếu đòn bẩy chi phí
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — kinh điển RAG đánh giá framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - observability trôi tham chiếu
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) - Bộ phân loại an toàn 2026
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) - framework đường sắt policy
