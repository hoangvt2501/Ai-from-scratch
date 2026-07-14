# Capstone 05 — Agent nghiên cứu tự trị (AI-Scientist Class)

> AI-Scientist-v2 của Sakana đã xuất bản toàn bộ các bài báo. Phòng thí nghiệm Agent đã thực hiện các thí nghiệm. Allen AI chia sẻ traces. Hình dạng năm 2026 là tìm kiếm cây lập kế hoạch-thực hiện-xác minh qua các thử nghiệm, chi phí ngân sách, thực thi mã sandbox, trình viết LaTeX phản hồi tầm nhìn và nhóm người đánh giá kiểu NeurIPS tự động. Điểm mấu chốt là xây dựng một cái, chạy nó từ đầu đến cuối trong vòng 30 đô la cho mỗi tờ báo và sống sót sau đội đỏ trốn thoát sandbox mà Sakana đã ghi lại.

**Loại:** Đá đỉnh
**Ngôn ngữ:** Python (agent + sandbox), LaTeX (output)
**Kiến thức tiên quyết:** Giai đoạn 2 (ML), Giai đoạn 3 (học sâu), Giai đoạn 7 (transformers), Giai đoạn 10 (LLMs từ đầu), Giai đoạn 14 (agents), Giai đoạn 15 (tự chủ), Giai đoạn 16 (đa agent), Giai đoạn 18 (an toàn)
**Các giai đoạn thực hiện: **P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · Trang 18
**Thời lượng:** 40 giờ

## Vấn đề

Nghiên cứu tự trị agents đã vượt qua ngưỡng vào năm 2026. AI-Scientist-v2 của Sakana AI đã được xuất bản trên tạp chí Nature với các bài báo được tạo ra đã vượt qua đánh giá ngang hàng của hội thảo. ShinkaEvolve (ICLR 2026) đã mở rộng dòng cho các giả thuyết đang phát triển. Phòng thí nghiệm Agent của AMD shipped traces có thể tái tạo. Các agents không phải là phép thuật - chúng là một vòng lặp kế hoạch-thực hiện-xác minh chạy trên một cây thử nghiệm ứng viên, với giới hạn chi phí, sandboxes ràng buộc hạt giống và đánh giá tự động. Thủ công nằm trong vòng lặp, ngân sách và câu chuyện an toàn.

Bạn học vòng lặp bằng cách thực hiện một vòng lặp chống lại một ý tưởng hạt giống trong một miền hẹp (ví dụ: cắt bỏ attention-thưa thớt trên 100M-parameter transformer). Giá trị không phải là khám phá điều gì đó mới trong lần chạy đầu tiên. Giá trị nằm trong cơ sở hạ tầng: tìm kiếm cây, sandbox thử nghiệm, vòng lặp người viết-người phản biện, báo cáo nhóm đỏ. Nhóm Sakana đã ghi lại những thất bại sandbox thoát hiểm; agent của bạn phải vượt qua cùng một đội màu đỏ.

## Khái niệm

agent là tìm kiếm cây đầu tiên tốt nhất. Các nút là thông số kỹ thuật thử nghiệm: (giả thuyết, config, mã, kết quả mong đợi). Một bước mở rộng đề xuất những đứa trẻ với các chỉnh sửa nhỏ (hoán đổi optimizer, thay đổi kích thước batch, xóa bỏ một thành phần). Mỗi đứa trẻ chạy trong một sandbox mới với giới hạn tài nguyên cứng. Kết quả được đưa trở lại chức năng tính điểm xếp hạng các nút theo (tính mới × chất lượng × ngân sách còn lại). Cây phát triển cho đến khi ngân sách cạn kiệt, sau đó branch tốt nhất được viết ra.

Người viết là đa phương thức. Nó tạo ra một bản nháp LaTeX, biên dịch nó, hiển thị các số liệu và đưa PDF được kết xuất trở lại chế độ tầm nhìn của Claude Opus 4.7 để phê bình về bố cục, tính dễ đọc của hình và alignment bằng chứng tuyên bố. Một nhóm đánh giá gồm năm giám khảo LLM phát ra điểm số theo phong cách NeurIPS (mới lạ, nghiêm ngặt, rõ ràng, khả năng tái tạo, tác động); Nếu mức trung bình giảm xuống dưới ngưỡng, bài báo sẽ trở lại với người viết với lời phê bình.

An toàn là chịu tải. Mọi thử nghiệm đều chạy trong sandbox E2B hoặc Daytona mà không có lối ra mạng, đồng hồ treo tường có giới hạn và giới hạn tài nguyên được ghim. Bước tạo mã của agent đi qua một lớp policy chặn các lệnh gọi hệ thống thoát khỏi sandbox. Báo cáo của đội đỏ tái tạo bề mặt tấn công do Sakana ghi lại (bom ngã ba, thoát khỏi hệ thống tập tin, LLM các cuộc gọi mạng được viết).

## Kiến trúc

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- Orchestration: LangGraph với cổng kiểm tra và cổng phê duyệt của con người
- Tìm kiếm cây: tùy chỉnh tốt nhất trước các nút thử nghiệm (kiểu AB-MCTS từ Sakana v2)
- Sandbox: E2B mỗi thử nghiệm, dự phòng Docker trong Docker; Giới hạn tài nguyên qua cGroups
- Văn học: Semantic Scholar Graph API + OpenAlex + bộ nhớ đệm FAISS cục bộ của các bản tóm tắt
- Người viết: Mẫu LaTeX + Claude Opus 4.7 (chế độ tầm nhìn) để phê bình và bố cục hình
- Người đánh giá: nhóm 5 giám khảo (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) với tổng hợp có trọng số
- Thử nghiệm framework: PyTorch 2.5 cho thí nghiệm vật lý, W & B cho ghi nhật ký
- Observability: Langfuse cho agent traces, ngân sách cứng 30 đô la cho mỗi tờ

## Tự xây dựng

1. **Phạm vi hạt giống và miền.** Lấy ý tưởng hạt giống (ví dụ: "điều tra các mô hình thưa thớt trong bản đồ attention của transformers dưới 1B"). Xác định không gian tìm kiếm: models, datasets, ngân sách tính toán.

2. **Văn học vượt qua.** Truy vấn Semantic Scholar + OpenAlex cho 50 bài báo liên quan được trích dẫn nhiều nhất; bộ nhớ cache tóm tắt cục bộ; Tạo thông báo tên miền dài 1 trang.

3. **Giàn giáo cây.** Khởi tạo gốc bằng giả thuyết hạt giống. Thực hiện `expand(node) -> children` với các đề xuất sửa đổi nhỏ (một config thay đổi cho mỗi đứa trẻ). Thực hiện `score(node)` như một tính mới có trọng số × chất lượng × thuật ngữ ngân sách.

4. **Sandbox bao bọc.** Mọi thử nghiệm đều chạy `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (hoặc policy E2B tương đương). Hạt giống được viết cho sandbox; Đầu ra được gắn trở lại ở chế độ chỉ đọc.

5. **Plan-execute-verify loop.** `plan` đề xuất con. `execute` chạy sandbox, ghi lại nhật ký và số liệu. `verify` chạy kiểm tra đơn vị trên các chỉ số (loss có giảm không? việc cắt bỏ có cô lập hiệu ứng không?). Các nút bị lỗi nhận được lý do lỗi được lưu trữ trên cây.

6. **Nhà văn.** Sau ngân sách, hãy chọn branch tốt nhất. Hiển thị số liệu với matplotlib. Tạo bản nháp LaTeX thông qua Claude Opus 4.7 với branch trace trong ngữ cảnh. Biên dịch. Cung cấp PDF đã biên dịch trở lại tầm nhìn Opus 4.7 để phê bình. Lặp lại.

7. **Nhóm người đánh giá.** Năm giám khảo chấm điểm bản nháp (tính mới, nghiêm ngặt, rõ ràng, khả năng tái tạo, tác động) với các bảng đánh giá kiểu NeurIPS. Nếu có ý nghĩa < 4.0/5, hãy quay trở lại với nhà văn với lời phê bình. Dừng cứng sau 3 lần viết lại.

8. **Đội đỏ.** Xây dựng hoặc tích hợp một tập hợp các nhiệm vụ đối nghịch nhắm vào sandbox: bom ngã ba, nỗ lực lấy cắp mạng, thoát khỏi hệ thống tệp, siêu ký tự shell LLM viết. Xác nhận tất cả đều bị chặn. Viết ra những phát hiện.

9. **Khả năng tái tạo.** Mỗi bài báo đều ships với trace JSON tìm kiếm cây, hạt giống, liên kết chạy W & B, cấu hình sandbox và README tái tạo nó từ đầu đến cuối.

## Ứng dụng

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Sản phẩm bàn giao

`outputs/skill-ai-scientist.md` là sản phẩm được giao. Với một ý tưởng hạt giống + một tên miền + ngân sách 30 đô la, nó chạy toàn bộ pipeline và phát hành một bài báo có thể xem xét cộng với một gói khả năng tái tạo.

| Trọng lượng | Tiêu chí | Cách đo lường |
| :-: | --- | --- |
| 25 | Chất lượng giấy | Đánh giá đánh giá mù đối với các bài báo hội thảo đã xuất bản |
| 20 | Nghiêm ngặt thử nghiệm | Đường cơ sở, hạt giống, cắt bỏ; mọi tuyên bố được hỗ trợ bởi một ô trong bảng kết quả |
| 20 | Kỷ luật chi phí và tính toán | $30/paper trần thực thi, Langfuse-traced |
| 20 | Sự An Toàn | Sandbox đội đỏ vượt qua; Đã xác minh policy mạng và ngắt kết nối |
| 15 | Khả năng tái tạo | Chạy lại một lệnh với các hạt giống giống hệt nhau sẽ tái tạo giấy |
| **100** |||

## Bài tập

1. Chạy pipeline dựa trên ba ý tưởng hạt giống khác nhau trong cùng một miền. So sánh phần nào của tìm kiếm cây chồng chéo. Xác định điện toán lãng phí trùng lặp.

2. Thêm cổng con người trong vòng lặp trước khi thực hiện thử nghiệm cho các nút ước tính trên 5 đô la. Đo lường tổng chi phí giảm bao nhiêu.

3. Hoán đổi nhóm người phản biện cho một giám khảo duy nhất. Đo lường tỷ lệ chấp nhận sai trên một tập hợp các giấy tờ xấu đã biết.

4. Giới thiệu kiểm tra nhóm đỏ đánh cắp mạng: agent viết mã cố gắng `curl` một địa chỉ bên ngoài. Xác nhận `--network=none` policy chặn nó. Ghi lại nỗ lực.

5. So sánh tìm kiếm cây của bạn với đường cơ sở ngẫu nhiên cố định (cùng ngân sách, không có chiến lược mở rộng). Báo cáo tính mới × tăng chất lượng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Tìm kiếm cây | "Mở rộng kiểu AB-MCTS" | Khám phá ưu tiên tốt nhất so với các nút thử nghiệm với điểm mới×chất lượng×ngân sách |
| Sandbox | "Cô lập thí nghiệm" | Container không có mạng, CPU/memory có giới hạn, hạt giống được ghim, đầu vào chỉ đọc |
| Phê bình tầm nhìn | "Kết xuất sau đó đọc" | Biên dịch bài báo thành PDF, đưa PDF trở lại VLM để phê bình bố cục và bằng chứng tuyên bố |
| Nhóm phản biện | "Đánh giá ngang hàng tự động" | Nhiều giám khảo LLM chấm điểm bài báo bằng bảng đánh giá NeurIPS; cổng cốt liệu có trọng lượng pipeline |
| Điểm mới lạ | "Cái này có mới không?" | Heuristic phạt sự gần gũi với bộ nhớ đệm tài liệu 50 bài |
| Trần chi phí | "$ ngân sách" | Giới hạn cứng trên tổng chi tiêu cho mỗi bài báo; Bộ đếm Langfuse + ước tính trước khi chạy |
| Đội đỏ | "Kiểm toán Sandbox thoát" | Các nhiệm vụ đối nghịch sẽ thoát khỏi sandbox nếu policy sai |

## Đọc thêm

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) - tài liệu tham khảo production nghiên cứu agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — phương pháp luận ban đầu
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) - mở rộng tiến hóa
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — framework phòng thí nghiệm nghiên cứu đa vai trò
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — lớp orchestration tham chiếu
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — tìm kiếm tài liệu
- [E2B sandboxes](https://e2b.dev) - cách ly thí nghiệm tham chiếu
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — bảng đánh giá mà nhóm người đánh giá mã hóa
