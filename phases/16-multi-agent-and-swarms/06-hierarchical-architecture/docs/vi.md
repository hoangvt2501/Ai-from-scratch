# Kiến trúc phân cấp và chế độ lỗi của nó

> Phân cấp là người giám sát lồng nhau. Người quản lý agents hơn các nhà quản lý phụ trên workers. CrewAI `Process.hierarchical` là phiên bản sách giáo khoa: một `manager_llm` tự động ủy quyền nhiệm vụ và xác thực đầu ra. Tương đương với LangGraph là `create_supervisor(create_supervisor(...))`. Đó là mô hình tự nhiên khi nhiệm vụ là một sơ đồ tổ chức thực sự. Đây cũng là mô hình có nhiều khả năng sụp đổ thành vòng lặp quản lý - người quản lý agents phân công công việc kém, hiểu sai kết quả phụ hoặc không đạt được sự đồng thuận. Tuần tự thường đánh bại nó.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 05 (Mẫu giám sát)
**Thời lượng:** ~60 phút

## Vấn đề

Một khi mẫu người giám sát nhấp chuột, bước tiếp theo tự nhiên là "điều gì sẽ xảy ra nếu workers họ là người giám sát?" Các đội có các đội phụ; công ty có các phòng ban của các phòng ban. Kiến trúc phân cấp phản ánh điều đó.

Vấn đề: các nhà quản lý LLM không giống như các nhà quản lý con người. Một người quản lý có priors ổn định về những gì báo cáo của họ biết. Một người quản lý LLM lý luận lại tổ chức mỗi lần từ bất cứ điều gì trong bối cảnh của nó. Trôi dạt nhỏ trong bối cảnh đó, và toàn bộ cái cây phân bổ sai công việc.

## Khái niệm

### Hình dạng

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Mọi nút nội bộ lập kế hoạch, ủy quyền và tổng hợp. Chỉ có lá mới có tác dụng.

### Nơi nó tỏa sáng

- **Rõ ràng ánh xạ tổ chức.** Nếu nhiệm vụ thực sự là bộ phận ("xem xét pháp lý tài liệu, tài chính xem xét tài liệu, kỹ thuật xem xét tài liệu, sau đó tóm tắt cho giám đốc điều hành"), hệ thống phân cấp là rõ ràng.
- **Tóm tắt cục bộ.** Mỗi người quản lý phụ tổng hợp đầu ra của nhóm trước khi người quản lý cấp cao nhìn thấy. Người quản lý hàng đầu nhìn thấy ba bản tóm tắt của người quản lý phụ, không phải mười lăm worker đầu ra.

### Nơi nó bị hỏng

Ba chế độ thất bại mà các cuộc khám nghiệm tử thi năm 2026 tiếp tục tìm thấy:

1. **Lỗi phân công nhiệm vụ.** Người quản lý đọc mục tiêu, ảo giác về sự phân rã và ủy quyền cho người quản lý phụ sai. Bởi vì người quản lý phụ ngoan ngoãn làm việc trên những gì nó được đưa ra, lỗi chỉ xuất hiện ở tổng hợp trên cùng - một cấp độ khác với nơi con người có thể bắt được nó.
2. **Hiểu sai đầu ra.** Người quản lý phụ trả về "không thể xác minh xác nhận quyền sở hữu X". Người quản lý hàng đầu tóm tắt là "tuyên bố X chưa được xác nhận". Ý nghĩa trôi dạt ở mọi cấp độ.
3. **Vòng lặp đồng thuận.** Hai người quản lý phụ không đồng ý; quản lý hàng đầu yêu cầu họ hòa giải; họ ủy quyền lại; workers chạy lại; các nhà quản lý phụ trả về câu trả lời hơi khác nhau; vòng lặp. `Process.hierarchical` của CrewAI bảo vệ chống lại điều này bằng giới hạn bước, nhưng bản thân giới hạn hiện là một hyperparameter.

### Câu hỏi quyết định

Tuần tự (pipeline tuyến tính) so với phân cấp: nhiệm vụ của bạn có thực sự có các nhóm con độc lập hay đó là một luồng tuyến tính giả vờ là một cái cây? Nếu sau này, hãy sử dụng tuần tự. Nếu trước đây, hãy sử dụng các quy tắc đối chiếu phân cấp nhưng rõ ràng về ngân sách.

### Triển khai của CrewAI

`Process.hierarchical` gọi điện cho một người quản lý LLM các đội chuyên môn. Người quản lý:

- nhận nhiệm vụ cấp cao nhất,
- giao nhiệm vụ phụ cho các nhóm,
- đánh giá kết quả của phi hành đoàn,
- quyết định chấp nhận, ủy quyền lại hay lặp lại.

Tài liệu: https://docs.crewai.com/en/introduction (tìm "Process phân cấp" trong Khái niệm cốt lõi).

### Triển khai LangGraph

LangGraph sử dụng các lệnh gọi `create_supervisor` lồng nhau. Người giám sát bên trong có biểu đồ riêng; Người giám sát bên ngoài coi đồ thị bên trong như một nút mờ đục. Điều này sạch hơn CrewAI để gỡ lỗi (bạn có thể bước qua từng biểu đồ riêng biệt) nhưng khó thể hiện việc định hình lại động của cây.

Tham khảo: https://reference.langchain.com/python/langgraph-supervisor.

## Tự xây dựng

`code/main.py` chạy hệ thống phân cấp 3 cấp:

- quản lý hàng đầu: chia nhiệm vụ thành branches "kỹ thuật" và "pháp lý",
- Quản lý phụ kỹ thuật: chia thành workers "frontend" và "backend",
- Giám đốc phụ pháp lý: một worker.

Demo đối chiếu con đường hạnh phúc (mọi người đều đồng ý) với một **con đường hỗn loạn** nơi sự phân rã của người quản lý cấp cao dán nhãn sai "hợp pháp" là "tài chính" và theo dõi dòng thác lỗi - người quản lý phụ ngoan ngoãn làm công việc tài chính, người tổng hợp hàng đầu báo cáo kết quả tài chính, câu hỏi pháp lý ban đầu không được trả lời.

Chạy:

```
python3 code/main.py
```

Đầu ra hiển thị cả hai đường dẫn với sự cạnh tranh rõ ràng giữa "những gì đã được yêu cầu" và "những gì đã được giao".

## Ứng dụng

`outputs/skill-hierarchy-fitness.md` đánh giá xem một nhiệm vụ nhất định nên sử dụng trình giám sát phân cấp, tuần tự hay phẳng. Đầu vào: mô tả nhiệm vụ, cấu trúc tổ chức, ngân sách đối chiếu. Đầu ra: đề xuất mẫu với các chế độ lỗi cụ thể để bảo vệ.

## Sản phẩm bàn giao

Nếu bạn ship phân cấp:

- **Giới hạn độ sâu cây ở mức 2.** Ba cấp độ đã che giấu hầu hết các lỗi khỏi observability.
- **Ngân sách đối chiếu rõ ràng.** Đặt vòng tối đa trước khi người quản lý cấp cao phải commit. Thường là 2.
- **Nguồn gốc trên mọi tổng hợp.** Bản tóm tắt của mỗi nút phải trích dẫn đầu ra lá nào đã tạo ra nó.
- **Cảnh báo về sự trôi dạt phân hủy.** Ghi lại sự phân hủy của người quản lý trên mỗi bước; diff so với truy vấn của người dùng. Nếu quá trình phân tách không còn bao gồm truy vấn, hãy kích hoạt cảnh báo.

## Bài tập

1. Chạy `code/main.py` và so sánh hạnh phúc và rối loạn. Cần bao nhiêu cấp độ chuyển giao người quản lý trước khi đầu ra hàng đầu hoàn toàn khác với câu hỏi của người dùng?
2. Thêm cấp độ thứ ba (trên cùng → phụ → → worker phụ phụ). Đo tần suất đường bị nhiễu loạn tự điều chỉnh so với phân kỳ hoàn toàn khi độ sâu tăng lên.
3. Triển khai worker "canary" tại mỗi người quản lý phụ luôn được hỏi câu hỏi ban đầu của người dùng không thay đổi. Sử dụng câu trả lời canary để phát hiện trôi phân hủy. Nhà quản lý nên phản ứng như thế nào khi canary không đồng ý với câu trả lời tổng hợp?
4. Đọc tài liệu `Process.hierarchical` của CrewAI. Xác định một guardrail cụ thể mà CrewAI áp dụng (giới hạn bước, manager_llm ràng buộc) và mô tả chế độ thất bại mà nó nhắm mục tiêu.
5. So sánh các trình giám sát LangGraph lồng nhau với hệ thống phân cấp CrewAI. Điều gì làm cho các vòng lặp đối chiếu rẻ hơn để phát hiện?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Phân cấp | "Mô hình sơ đồ tổ chức" | Giám sát viên đối với người giám sát; chỉ có lá mới hoạt động. |
| Quản lý LLM | "Ông chủ" | LLM phân tách, gán và xác thực tại một nút nội bộ. |
| Trôi phân hủy | "Ông chủ thua cốt truyện" | Sự chia tay của người quản lý hàng đầu không còn bao gồm câu hỏi ban đầu. |
| Vòng lặp đối chiếu | "Cuộc họp bất tận" | Người quản lý phụ không đồng ý; đại biểu lại hàng đầu; workers chạy lại; vòng lặp cho đến khi hết ngân sách. |
| Trần Depth-2 | "Đừng đi sâu hơn 2 cấp độ" | guardrail thực nghiệm: 3+ cấp độ sụp đổ observability. |
| Câu hỏi Canary | "Ground truth ở mọi cấp độ" | Một worker luôn được yêu cầu truy vấn ban đầu không thay đổi, để phát hiện độ trôi. |
| Chuỗi xuất xứ | "Ai nói gì" | Trace từ mỗi lần tổng hợp trở lại đầu ra của lá đã tạo ra nó. |

## Đọc thêm

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — phân cấp sách giáo khoa với người quản lý LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — người giám sát lồng nhau thông qua `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) - tại sao Anthropic cố tình chọn người giám sát phẳng thay vì phân cấp
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - phân loại MAT; Phần về Thất bại phối hợp Tài liệu Phân hủy trôi dạt
