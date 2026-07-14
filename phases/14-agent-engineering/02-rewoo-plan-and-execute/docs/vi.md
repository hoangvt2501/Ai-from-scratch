# ReWOO và Lập kế hoạch và thực hiện: Lập kế hoạch tách rời

> ReAct xen kẽ suy nghĩ và hành động trong một luồng. ReWOO tách chúng ra: một kế hoạch lớn trước, sau đó thực hiện. tokens ít hơn 5 lần, accuracy +4% trên HotpotQA và bạn có thể chắt lọc công cụ lập kế hoạch thành model 7B. Lập kế hoạch và thực hiện khái quát hóa nó; Plan-and-Act đã mở rộng quy mô nó thành điều hướng web.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Agent vòng lặp)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao phân tách Planner / Worker / Solver của ReWOO tiết kiệm tokens và cải thiện độ mạnh mẽ so với vòng lặp xen kẽ của ReAct.
- Triển khai DAG kế hoạch, trình thực thi theo thứ tự phụ thuộc và trình giải tổng hợp worker đầu ra — tất cả stdlib.
- Quyết định khi nào một nhiệm vụ nên chạy theo kế hoạch sau đó thực hiện so với ReAct xen kẽ, sử dụng khung "năm mẫu quy trình làm việc" (Anthropic) năm 2026.
- Nhận biết khi nào dữ liệu kế hoạch tổng hợp của Plan-and-Act là cần thiết cho các tác vụ web hoặc di động có đường chân trời dài.

## Vấn đề

Vòng lặp suy nghĩ-hành động-quan sát xen kẽ của ReAct rất đơn giản và linh hoạt, nhưng mỗi lệnh gọi công cụ phải mang đầy đủ ngữ cảnh prior - bao gồm mọi suy nghĩ trước đó. Token sử dụng tăng lên bậc hai theo chiều sâu. Tệ hơn: khi một công cụ bị lỗi giữa vòng lặp, model phải lấy lại toàn bộ kế hoạch từ việc quan sát lỗi.

ReWOO (Xu và cộng sự, arXiv:2305.18323, Tháng Năm 2023) nhận thấy điều này và đặt cược: lập kế hoạch trước toàn bộ mọi thứ, lấy bằng chứng song song, soạn câu trả lời ở cuối. Một LLM kêu gọi lập kế hoạch, N công cụ kêu gọi bằng chứng (có thể song song), một LLM kêu gọi giải quyết. Giao dịch kém linh hoạt hơn (kế hoạch tĩnh) để có hiệu quả token tốt hơn nhiều và các chế độ lỗi rõ ràng hơn.

## Khái niệm

### Ba vai trò

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner tạo ra một DAG. Mỗi nút đặt tên cho một công cụ, các đối số của nó và nó phụ thuộc vào các nút trước đó (các tham chiếu như `#E1`, `#E2`). Workers thực thi các nút theo thứ tự tô pô. Solver khâu mọi thứ lại với nhau.

### Tại sao ít tokens hơn 5 lần

ReAct tăng prompt độ dài tuyến tính với số bước. Ở bước 10, prompt chứa suy nghĩ 1 cộng với hành động 1 cộng với quan sát 1 cộng với suy nghĩ 2 cộng với hành động 2 cộng với quan sát 2, v.v. Mỗi bước trung gian cũng bao gồm các prompt ban đầu một cách dư thừa.

ReWOO trả một prompt lập kế hoạch (lớn), N worker prompts nhỏ (mỗi người chỉ gọi công cụ, không có chuỗi) và một prompt trình giải quyết. Trên HotpotQA, giấy đo tokens ít hơn ~5 lần trong khi đạt +4 accuracy tuyệt đối.

### Tại sao nó mạnh mẽ hơn

Nếu worker 3 không thành công trong ReAct, vòng lặp phải lý do lỗi giữa luồng. Trong ReWOO, worker 3 trả về một chuỗi lỗi; Trình giải xem nó trong ngữ cảnh với kế hoạch ban đầu và có thể xuống cấp một cách duyên dáng. Bản địa hóa lỗi là trên mỗi nút, không phải theo bước.

### Lập kế hoạch distillation

Kết quả thứ hai của bài báo: vì người lập kế hoạch không nhìn thấy các quan sát, bạn có thể fine-tune model 7B về kết quả của công cụ lập kế hoạch từ một giáo viên 175B. model nhỏ xử lý kế hoạch; model lớn không cần thiết ở inference. Điều này hiện là tiêu chuẩn - nhiều production agents 2026 sử dụng một công cụ lập kế hoạch nhỏ và một người thực thi lớn hoặc ngược lại.

### Lập kế hoạch và thực hiện (LangChain, 2023)

Bài đăng vào tháng 8 năm 2023 của nhóm LangChain đã khái quát hóa ReWOO thành một tên mẫu: Lập kế hoạch và thực hiện. Công cụ lập kế hoạch trước phát ra danh sách bước, trình thực thi chạy từng bước, công cụ lập kế hoạch lại tùy chọn có thể sửa đổi sau khi quan sát kết quả. Điều này gần với ReAct hơn là ReWOO (công cụ lập kế hoạch lại đưa các quan sát trở lại lập kế hoạch) nhưng bảo toàn khoản tiết kiệm token.

### Kế hoạch và Hành động (Erdogan và cộng sự, arXiv: 2503.09572, ICML 2025)

Plan-and-Act mở rộng mô hình thành agents web và di động có đường chân trời dài. Đóng góp chính là dữ liệu kế hoạch tổng hợp: một trình tạo quỹ đạo được gắn nhãn tạo ra dữ liệu training nơi kế hoạch rõ ràng. Được sử dụng để fine-tune models lập kế hoạch tiếp tục làm việc quá 30–50 bước trên các nhiệm vụ giống như WebArena, trong đó một quỹ đạo ReAct duy nhất mất tính mạch lạc.

### Khi nào nên chọn cái nào

| Mô hình | Khi nào |
|---------|------|
| Hành động lại | Nhiệm vụ ngắn, môi trường không xác định, cần xử lý ngoại lệ phản ứng |
| REWOO | Các nhiệm vụ có cấu trúc với các công cụ đã biết, bằng chứng nhạy cảm token, có thể song song |
| Lập kế hoạch và thực hiện | Giống như ReWOO nhưng có quy hoạch lại sau khi thực hiện một phần |
| Lập kế hoạch và hành động | Đường chân trời dài (>30 bước), web/mobile/computer-use |
| Cây tư tưởng | Tìm kiếm đáng để trả tiền (Bài 04) |

Hướng dẫn tháng 12 năm 2024 của Anthropic: bắt đầu với những điều đơn giản nhất. Nếu nhiệm vụ là một lệnh gọi công cụ cộng với một bản tóm tắt, đừng xây dựng ReWOO. Nếu nhiệm vụ là một nhiệm vụ nghiên cứu gồm 40 bước, đừng thực hiện ReAct một mình.

## Tự xây dựng

`code/main.py` triển khai đồ chơi ReWOO:

- `Planner` — một policy có kịch bản phát ra DAG kế hoạch từ một prompt.
- `Worker` — gửi lệnh gọi công cụ của từng nút thông qua registry.
- `Solver` - bố cục theo kịch bản đọc bằng chứng và đưa ra câu trả lời cuối cùng.
- Độ phân giải phụ thuộc — các tham chiếu như `#E1` được thay thế bằng các đầu ra worker trước đó.

Bản demo trả lời "Dân số của thủ đô nước Pháp, làm tròn đến hàng triệu là bao nhiêu?" bằng cách sử dụng kế hoạch hai bước: (1) tra cứu thủ đô, (2) tra cứu dân số, sau đó giải quyết.

Chạy nó:

```
python3 code/main.py
```

trace hiển thị kế hoạch đầy đủ trước, sau đó worker kết quả, sau đó là thành phần giải. So sánh số lượng token (chúng tôi in số ký tự sơ bộ) với chạy xen kẽ kiểu ReAct - ReWOO giành chiến thắng trong loại nhiệm vụ có cấu trúc này.

## Ứng dụng

LangGraph ships Plan-and-Execute như một công thức (`create_react_agent` cho ReAct, biểu đồ tùy chỉnh cho plan-execute). Flows của CrewAI mã hóa mẫu trực tiếp: bạn xác định các tác vụ trước và Flow DAG thực thi chúng. Phương pháp tiếp cận dữ liệu tổng hợp của Plan-and-Act vẫn chủ yếu là nghiên cứu; mẫu runtime (DAG kế hoạch rõ ràng) ships trong production thông qua LangGraph và CrewAI Flows.

## Sản phẩm bàn giao

`outputs/skill-rewoo-planner.md` tạo DAG kế hoạch ReWOO từ yêu cầu của người dùng, được cung cấp một danh mục công cụ. Nó xác thực kế hoạch (không tuần hoàn, mọi tham chiếu đã được giải quyết, mọi công cụ tồn tại) trước khi chuyển giao cho người thực thi.

## Bài tập

1. Song song hóa việc thực thi worker cho các nút kế hoạch độc lập. Nó mua cho bạn điều gì trên DAG 6 nút với 2 nhóm song song?
2. Thêm nút lập kế hoạch lại kích hoạt nếu có bất kỳ worker nào trả về lỗi. Thay đổi nhỏ nhất đối với ReWOO khiến nó trở thành Lập kế hoạch và Thực hiện là gì?
3. Thay thế `Planner` bằng một model nhỏ (7B class) và giữ `Solver` trên model biên giới. So sánh chất lượng từ đầu đến cuối — phân tách thất bại ở đâu?
4. Đọc Phần 4 của bài báo ReWOO về distillation lập kế hoạch. Tái tạo kết quả 175B -> 7B một cách khái niệm: bạn cần dữ liệu training gì và làm thế nào để bạn chấm điểm chất lượng kế hoạch?
5. Chuyển đồ chơi vào hình dạng quỹ đạo của Plan-and-Act: kế hoạch là một trình tự, không phải DAG. Sự đánh đổi nào thay đổi?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| REWOO | "Lý luận mà không cần quan sát" | Lập kế hoạch, sau đó lấy bằng chứng song song, sau đó giải quyết - không có quan sát nào trong prompt lập kế hoạch |
| Lập kế hoạch và thực hiện | "Mô hình thực hiện kế hoạch của LangChain" | ReWOO với một nút lập kế hoạch lại tùy chọn sau khi thực hiện |
| Lập kế hoạch và hành động | "Kế hoạch thực hiện theo quy mô" | Phân chia planner/executor rõ ràng với kế hoạch tổng hợp training dữ liệu cho các nhiệm vụ dài hạn |
| Tham khảo bằng chứng | "#E1, #E2, ..." | Trình giữ chỗ nút kế hoạch được thay thế bằng đầu ra prior worker tại thời điểm điều phối |
| Lập kế hoạch distillation | "Người lập kế hoạch nhỏ, người thực hiện lớn" | Fine-tune một model nhỏ về kế hoạch traces từ một giáo viên lớn |
| Token hiệu quả | "Ít chuyến đi khứ hồi hơn" | tokens ít hơn 5 lần trên HotpotQA so với ReAct trên báo |
| Trình thực thi DAG | "Người điều phối cấu trúc liên kết" | Chạy các nút kế hoạch theo thứ tự phụ thuộc; song song ở mỗi cấp độ |

## Đọc thêm

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — bài báo kinh điển
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — người lập kế hoạch-thực thi theo quy mô với các kế hoạch tổng hợp
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) - công thức framework
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - chọn mẫu đơn giản nhất hoạt động
