# Mô hình giám sát / Orchestrator-Worker

> Một lãnh đạo agent kế hoạch và đại biểu; chuyên biệt workers thực hiện trong ngữ cảnh song song và báo cáo lại. Đây là mô hình đằng sau hệ thống Nghiên cứu của Anthropic (Claude Opus 4 là chì, Sonnet 4 là subagents), được đo ở mức +90,2% so với Opus 4 một agent trên các đánh giá nghiên cứu nội bộ. Bài đăng kỹ thuật của Anthropic báo cáo rằng 80% variance trên BrowseComp được giải thích chỉ bằng cách sử dụng token - nhiều agent chiến thắng phần lớn vì mỗi subagent đều có một context window mới. Bài học này xây dựng mô hình giám sát từ primitives và bao gồm các bài học kỹ thuật năm 2026 từ các triển khai production.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib, `threading`)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~75 phút

## Vấn đề

Nghiên cứu là nhiệm vụ nguyên mẫu mà các hệ thống một agent thất bại. Bạn hỏi "điều gì đã thay đổi trong các hệ thống đa agent từ năm 2023 đến năm 2026?" Một agent đọc năm bài báo tuần tự, lấp đầy một nửa ngữ cảnh của nó bằng văn bản của họ, và sau đó phải suy luận về tất cả chúng cùng nhau. Nó quên bài đầu tiên khi đến bài thứ năm. Nó không thể song song.

Mô hình giám sát khắc phục điều này: một khách hàng tiềm năng agent lập kế hoạch tìm kiếm, ủy quyền từng câu hỏi phụ cho một worker và tổng hợp. Mỗi worker có cửa sổ 200k token riêng cho một câu hỏi hẹp. Người dẫn đầu không bao giờ nhìn thấy các giấy tờ thô - chỉ nhìn thấy các bản tóm tắt worker.

Hệ thống Nghiên cứu production của Anthropic báo cáo +90,2% về đánh giá nghiên cứu nội bộ so với một Opus 4 duy nhất. Bài đăng tương tự lưu ý rằng 80% variance BrowseComp được giải thích bởi *token cách sử dụng*. Bối cảnh mới mỗi subagent là cơ chế chính.

## Khái niệm

### Mô hình

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Chì không bao giờ đọc nguyên liệu thô. Các workers không bao giờ nhìn thấy tác phẩm của nhau cho đến khi khách hàng tiềm năng tổng hợp. Mỗi mũi tên là một bàn giao với một artifact hẹp.

### Tại sao nó chiến thắng

Ba cơ chế:

1. **Bối cảnh mới mỗi subagent.** Một worker khám phá "di sản FIPA-ACL" không mang 40k tokens kế hoạch chi tiêu chính. Nó nhận được một cửa sổ 200k cho một câu hỏi.
2. **Chuyên môn hóa thông qua prompt.** prompt của khách hàng tiềm năng là "phân hủy và tổng hợp", không phải "nghiên cứu". prompt của mỗi worker đều hẹp: "tìm những gì đã thay đổi trong X". Tập trung prompts tạo ra đầu ra tập trung.
3. **Parallelism.** Workers chạy đồng thời. Thời gian đồng hồ treo tường gần như `max(worker_times) + plan + synthesis`, không phải `sum(worker_times)`.

### Bài học kỹ thuật (Anthropic 2025)

Bài đăng Anthropic liệt kê một số bài học production vẫn còn phù hợp với năm 2026:

- **Mở rộng nỗ lực truy vấn độ phức tạp.** Truy vấn đơn giản: một agent, 3-10 lệnh gọi công cụ. Truy vấn phức tạp: 10+ agents. Khách hàng tiềm năng phải ước tính điều này, không phải người gọi.
- **Rộng sau đó thu hẹp.** Trước tiên, hãy phân tách thành các câu hỏi phụ rộng, sau đó tạo ra nhiều workers hơn cho mỗi câu hỏi phụ nếu câu trả lời đảm bảo chiều sâu.
- **Triển khai cầu vồng.** Agents chạy lâu dài và có trạng thái. Màu xanh lam truyền thống không hoạt động. Anthropic sử dụng cầu vồng: rollout dần các phiên bản mới trong khi các phiên bản cũ cạn kiệt.
- **Token cách sử dụng chiếm ưu thế.** Multi-agent là ~15× tokens của một agent. Chỉ chạy nó khi giá trị nhiệm vụ phù hợp với chi phí.

### Bước ngoặt của LangGraph

LangGraph ban đầu shipped một thư viện `langgraph-supervisor` với trình trợ giúp `create_supervisor` cấp cao. Vào năm 2025, LangChain đã chuyển đề xuất sang triển khai mẫu giám sát thông qua gọi công cụ trực tiếp, vì các lệnh gọi công cụ cho phép kiểm soát nhiều hơn đối với *những gì người giám sát nhìn thấy* (kỹ thuật ngữ cảnh). Thư viện vẫn hoạt động; Các tài liệu hiện đề xuất biểu mẫu gọi công cụ.

### Các chế độ thất bại

- **Chì ảo giác kế hoạch.** Nếu khách hàng tiềm năng tạo ra các câu hỏi phụ không phân tích câu hỏi thực sự, workers thực hiện nghiên cứu chính xác về mục tiêu sai.
- **Workers khám phá quá mức.** Nếu không có ranh giới phạm vi rõ ràng, workers trôi dạt ra ngoài câu hỏi phụ được giao và làm ô nhiễm bước tổng hợp.
- **Xung đột tổng hợp.** Hai workers trả về các sự kiện mâu thuẫn. Khách hàng tiềm năng phải hỏi lại (thêm một vòng) hoặc ghi nhận sự bất đồng một cách rõ ràng. Im lặng chọn một bên là thất bại tồi tệ nhất: người dùng không bao giờ biết bất đồng đã xảy ra.

### Khi người giám sát sai

- **Nhiệm vụ tuần tự.** Nếu bước 2 thực sự cần đầu ra của bước 1, thì tính song song không mua được gì. Sử dụng pipeline (CrewAI Sequential, biểu đồ tuyến tính LangGraph).
- **Truy vấn đơn giản.** Single-agent xử lý chúng nhanh hơn và rẻ hơn. Sử dụng kiểm tra "nỗ lực mở rộng" của lead trước khi sinh workers.
- **Quyết định nghiêm ngặt.** Giám sát viên sử dụng ủy quyền do LLM chọn. Biểu đồ tĩnh tốt hơn khi audit/replay quan trọng hơn khả năng thích ứng.

```figure
supervisor-hierarchy
```

## Tự xây dựng

`code/main.py` thực hiện một giám sát ba workers song song bằng cách sử dụng `threading`. Khách hàng tiềm năng phân tách một truy vấn thành các câu hỏi phụ workers chạy đồng thời trên mỗi câu hỏi phụ và khách hàng tiềm năng tổng hợp. Không có LLMs thực sự - các workers được viết kịch bản để mô phỏng tìm nạp và tóm tắt.

Cấu trúc chính:

- `Lead.plan(query)` chia một truy vấn thành 3 câu hỏi phụ.
- `Worker.run(sub_q)` trả về một bản tóm tắt giả mạo (có thể là bất kỳ agent sử dụng công cụ nào trong production).
- `Lead.run(query)` bắt đầu workers trong threads, tham gia và tổng hợp.

Chạy:

```
python3 code/main.py
```

Đầu ra hiển thị kế hoạch, worker traces song song với dấu thời gian start/end và tổng hợp cuối cùng. Bạn có thể thấy đồng hồ treo tường chiến thắng: ba workers 0,3 giây chạy trong ~0,35 giây, không phải 0,9.

## Ứng dụng

`outputs/skill-supervisor-designer.md` lấy truy vấn của người dùng và tạo ra một thiết kế mẫu giám sát: system prompt dẫn đầu, vai trò worker, quy tắc phân tách câu hỏi phụ và mẫu tổng hợp. Sử dụng điều này trước khi xây dựng một hệ thống agent theo phong cách nghiên cứu mới.

## Sản phẩm bàn giao

Danh sách kiểm tra trước khi triển khai mẫu giám sát:

- **Model ghép nối.** Dẫn đầu trên một model cấp lý luận (Opus class, `o3` class). Workers trên một model nhanh hơn, rẻ hơn (Sonnet, `o4-mini`).
- **Worker timeout.** Bất kỳ worker nào vượt quá 2× runtime trung bình đều bị giết; Chì xuất hiện lại với phạm vi hẹp hơn hoặc tiếp tục mà không có phạm vi đó.
- **Token giới hạn cho mỗi worker.** Giới hạn cứng (giả sử 10× đầu vào tổng hợp dự kiến) ngăn chặn một worker chạy trốn thổi bay ngân sách.
- **Observability.** Trace kế hoạch của khách hàng tiềm năng, các lệnh gọi công cụ của mỗi worker và tổng hợp. Đây là cơ sở cho bất kỳ gỡ lỗi sau hoc nào.
- **Rainbow rollout.** agents chạy lâu dài có trạng thái cần chuyển đổi phiên bản dần dần, không phải hoán đổi nóng.

## Bài tập

1. Chạy `code/main.py`, sau đó sửa đổi dẫn đầu để sinh ra 5 workers thay vì 3. Quan sát hiệu ứng đồng hồ treo tường. Chi phí sinh sản vượt quá mức tiết kiệm song song trong bản demo này ở mức worker nào?
2. Thực hiện một worker timeout: giết bất kỳ worker nào chạy lâu hơn 0,5 giây và để khách hàng tiềm năng tổng hợp các kết quả còn lại. Bạn cần observability gì để biết một worker đã bị cắt?
3. Thêm một bước phát hiện xung đột vào tổng hợp của khách hàng tiềm năng: nếu hai workers trả về câu trả lời mâu thuẫn, khách hàng tiềm năng ghi nhận sự bất đồng thay vì chọn một. Làm thế nào để bạn phát hiện mâu thuẫn mà không cần gọi LLM?
4. Đọc bài viết Kỹ thuật hệ thống nghiên cứu của Anthropic. Liệt kê ba phương pháp mà bản demo đồ chơi này sẽ cần áp dụng để chạy trong production.
5. So sánh `create_supervisor` (kế thừa) của LangGraph với đề xuất gọi công cụ mới. Cái nào cho phép bạn kiểm soát tốt hơn những gì người giám sát nhìn thấy? Tại sao Anthropic rõ ràng chỉ chuyển các câu trả lời phụ chứ không phải ngữ cảnh worker thô vào tổng hợp?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Giám sát viên | "Dẫn agent" | Một người điều phối agent lập kế hoạch, ủy thác và tổng hợp. Không tự làm công việc. |
| Worker | "Subagent" | Một agent tập trung được người giám sát viện dẫn với phạm vi hẹp và context window riêng của nó. |
| Orchestrator-worker | "Mẫu giám sát" | Cùng một điều, tên khác nhau. Tài liệu năm 2026 sử dụng cả hai. |
| Bối cảnh mới mẻ | "Cửa sổ sạch" | Bối cảnh của một worker bắt đầu từ system prompt và câu hỏi được giao, không phải lịch sử của khách hàng tiềm năng. |
| Triển khai cầu vồng | "Dần dần rollout" | Các agents có trạng thái chạy lâu dài cần có phiên bản drain-and-replace, không phải xanh lam. |
| Token thống trị | "Bối cảnh là biến" | 80% variance nghiên cứu đến từ tổng số tokens được sử dụng, không phải model lựa chọn, mỗi Anthropic. |
| Mở rộng nỗ lực | "Khớp agent tính vào độ phức tạp" | Khách hàng tiềm năng ước tính độ khó của truy vấn, tạo ra 1 so với 10+ workers tương ứng. |
| Xung đột tổng hợp | "Workers không đồng ý" | Hai workers trả về các sự kiện mâu thuẫn; Người dẫn đầu phải nổi lên sự bất đồng, không được lặng lẽ chọn một. |

## Đọc thêm

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — tham chiếu production cho mẫu giám sát
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) - người giám sát gọi công cụ hiện là biểu mẫu được đề xuất
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) - người trợ giúp kế thừa, vẫn được sử dụng vào năm 2026 production
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — biến thể giám sát dựa trên bàn giao
