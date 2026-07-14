# Trò chuyện nhóm và lựa chọn diễn giả

> AutoGen GroupChat và AG2 GroupChat chia sẻ một cuộc trò chuyện trên N agents; Chức năng chọn (LLM, vòng tròn hoặc tùy chỉnh) chọn người phát biểu tiếp theo. Đây là nguyên mẫu của cuộc trò chuyện đa agent mới nổi - agents không biết vai trò của chúng trong biểu đồ tĩnh, họ chỉ phản ứng với nhóm được chia sẻ. Ngữ nghĩa GroupChat của AutoGen v0.2 được giữ nguyên trong nhánh AG2; AutoGen v0.4 đã viết lại nó như một diễn viên theo hướng sự kiện model. Microsoft đã đưa AutoGen vào chế độ bảo trì vào tháng 2 năm 2026 và merged nó với Semantic Kernel vào Microsoft Agent Framework (RC tháng 2 năm 2026). GroupChat primitive tồn tại ở cả AG2 và Microsoft Agent Framework - học một lần, sử dụng nó ở mọi nơi.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model)
**Thời lượng:** ~60 phút

## Vấn đề

Đồ thị tĩnh (LangGraph) rất tuyệt vời khi quy trình làm việc được biết đến. Các cuộc trò chuyện thực tế không tĩnh: đôi khi lập trình viên hỏi người đánh giá, đôi khi là nhà nghiên cứu, đôi khi là người viết. Mã hóa cứng mọi bàn giao có thể tạo ra một vụ nổ biên. Bạn muốn *agents phản ứng với một nhóm được chia sẻ*, với một số chức năng quyết định ai sẽ nói tiếp theo.

Đó chính xác là những gì AutoGen GroupChat làm.

## Khái niệm

### Hình dạng

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Mọi agent đều nhìn thấy mọi thông điệp. Một chức năng chọn được gọi ở mỗi lượt để chọn người nói tiếp theo.

### Ba hương vị chọn

**Vòng tròn.** Chu kỳ cố định. Xác định. Thang đo tuyến tính trong N nhưng bỏ qua ngữ cảnh - một lập trình viên có được lượt ngay cả khi chủ đề là xem xét pháp lý.

**Đã chọn LLM.** Một cuộc gọi đến một LLM đọc nhóm gần đây và trả về loa tiếp theo tốt nhất. Nhận biết ngữ cảnh nhưng chậm: mỗi lượt thêm một cuộc gọi LLM. Mặc định của AutoGen.

**Tùy chỉnh.** Một chức năng Python với bất kỳ logic nào bạn muốn. Điển hình: LLM-chọn với các quy tắc dự phòng (ví dụ: "luôn cho người xác minh lượt sau lập trình lập trình").

### The ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` giữ bộ chọn. Khi một agent hoàn thành một lượt, người quản lý sẽ gọi bộ chọn, bộ chọn này trả về agent tiếp theo. Vòng lặp tiếp tục cho đến khi có điều kiện kết thúc.

### Chấm dứt

Ba mẫu phổ biến:

- **Số vòng tối đa.** Giới hạn cứng trên tổng số lượt.
- **"TERMINATE" token.** Agents có thể phát ra một thông điệp canh gác; Người quản lý dừng lại khi một người xuất hiện.
- **Kiểm tra đạt mục tiêu.** Một trình xác minh nhẹ chạy mỗi lượt và dừng cuộc trò chuyện khi hoàn tất.

### AutoGen → AG2 tách ra và Microsoft Agent Framework merge

Vào đầu năm 2025, Microsoft bắt đầu viết lại AutoGen (v0.4) xung quanh một diễn viên hướng sự kiện model. Cộng đồng đã phân nhánh ngữ nghĩa GroupChat của AutoGen v0.2 thành AG2, giữ nguyên API mà những người sử dụng sớm đã tích hợp.

Vào tháng 2 năm 2026, Microsoft thông báo AutoGen sẽ chuyển sang chế độ bảo trì, với tác nhân theo hướng sự kiện model hợp nhất vào **Microsoft Agent Framework** (RC tháng 2 năm 2026, hiện merged với Semantic Kernel). Khái niệm GroupChat tồn tại trong cả hai bản nhạc; Các chi tiết triển khai khác nhau. AG2 là thượng nguồn ưu tiên cho mã tương thích với v0.2.

### Khi GroupChat phù hợp

- **Cuộc trò chuyện khẩn cấp.** Bạn không muốn nối trước mọi loa tiếp theo có thể.
- **Nhiệm vụ kết hợp vai trò.** Coder hỏi nhà nghiên cứu, nhà nghiên cứu hỏi người lưu trữ, người lưu trữ hỏi lại người lập trình. Flow không phải là DAG.
- **Giải quyết vấn đề thăm dò.** Hãy nghĩ đến "cuộc họp động não", không phải "dây chuyền lắp ráp".

### Khi nó không thành công

- **Quyết định luận nghiêm ngặt.** Bộ chọn LLM có thể không nhất quán. Cùng một prompt, chạy khác nhau, loa tiếp theo khác nhau.
- **Sycophancy xếp tầng.** Agents tuân theo bất kỳ ai nói một cách tự tin nhất. Phản prompt rõ ràng.
- **Bối cảnh cồng kềnh.** Mỗi agent đọc mọi tin nhắn; Sau 10 lượt, bối cảnh rất lớn. Sử dụng phép chiếu (Bài 15) để phạm vi chế độ xem.
- **Loa nóng.** Một agent thống trị cuộc trò chuyện vì người chọn ưu tiên các chuyên môn của nó. Giới thiệu cân bằng loa như một bộ chọn feature.

### Trò chuyện nhóm so với người giám sát

Cùng primitives, mặc định khác nhau:

- Người giám sát: một agent lập kế hoạch và những người khác thực hiện. Selector là "hỏi người lập kế hoạch phải làm gì".
- Trò chuyện nhóm: tất cả agents đều là đồng nghiệp; Selector là một chức năng trên nhóm được chia sẻ.

Cả hai đều sử dụng bốn primitives từ Bài 04. Trò chuyện nhóm mặc định ở trạng thái chia sẻ orchestration và toàn bộ nhóm do LLM chọn.

## Tự xây dựng

`code/main.py` triển khai GroupChat từ đầu trong stdlib. Ba agents (lập trình viên, người đánh giá, người quản lý), các biến thể vòng tròn và LLM chọn, và kết thúc trên `TERMINATE` token.

Bản demo in bản ghi cuộc trò chuyện cộng với trace quyết định của người chọn cho cả hai biến thể.

Chạy:

```
python3 code/main.py
```

## Ứng dụng

`outputs/skill-groupchat-selector.md` định cấu hình bộ chọn GroupChat cho một nhiệm vụ nhất định - vòng tròn so với LLM được chọn và tùy chỉnh và đầu vào bộ chọn nào (tin nhắn gần đây, agent chuyên môn, số lượt) để sử dụng.

## Sản phẩm bàn giao

Danh sách kiểm tra:

- **Giới hạn vòng tối đa.** Luôn luôn. 10-20 cho các nhiệm vụ điển hình.
- **Chỉ số cân bằng loa.** Theo dõi lượt trên mỗi agent; cảnh báo khi mất cân bằng vượt quá ngưỡng.
- **Chấm dứt token.** `TERMINATE` hoặc agent xác minh chuyên dụng.
- **Phép chiếu hoặc bộ nhớ có phạm vi.** Sau ~10 tin nhắn, hãy cân nhắc chỉ cung cấp cho mỗi agent một chế độ xem có phạm vi để tránh cồng kềnh ngữ cảnh.
- **Ghi nhật ký bộ chọn.** Đối với các biến thể được chọn LLM, hãy ghi nhật ký cả đầu vào của bộ chọn và lựa chọn của bộ chọn. Nếu không, việc gỡ lỗi là không thể.

## Bài tập

1. Chạy `code/main.py`. So sánh cuộc trò chuyện trong vòng tròn và LLM chọn. agent nào thống trị dưới mỗi loại?
2. Thêm quy tắc "max-speaks-per-agent" trong bộ chọn. Nó ảnh hưởng đến bảng điểm như thế nào?
3. Thực hiện chấm dứt đạt được mục tiêu: dừng khi người đánh giá trả về "đã được phê duyệt". Bao lâu thì nó trigger trước khi giới hạn tròn?
4. Đọc tài liệu ổn định AutoGen trên GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Xác định bộ chọn mặc định được `GroupChatManager` sử dụng.
5. Đọc AG2 repo (https://github.com/ag2ai/ag2) và so sánh GroupChat v0.2 của nó với phiên bản hướng sự kiện v0.4. Phiên bản 0.4 bổ sung thuộc tính cụ thể nào (thông lượng, khả năng chịu lỗi, khả năng kết hợp)?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Trò chuyện nhóm | "Agents trong một phòng trò chuyện" | Nhóm tin nhắn được chia sẻ + chức năng chọn. primitive AutoGen / AG2. |
| Lựa chọn loa | "Ai nói tiếp theo" | Chức năng chọn agent tiếp theo. Vòng tròn, LLM chọn hoặc tùy chỉnh. |
| GroupChatManager | "Người tổ chức cuộc họp" | AutoGen sở hữu bộ chọn và lặp lại qua các lượt. |
| Đại lý có thể trò chuyện | "Cơ sở agent" | class cơ sở AutoGen; một agent có thể gửi và nhận tin nhắn. |
| Chấm dứt token | "Từ 'dừng lại'" | Chuỗi Sentinel (thường là `TERMINATE`) kết thúc cuộc trò chuyện. |
| Loa nóng | "Một agent thống trị" | Chế độ lỗi trong đó bộ chọn tiếp tục chọn cùng một agent. |
| Bối cảnh phình to | "Hồ bơi phát triển không giới hạn" | Mỗi agent đọc mọi tin nhắn prior; bối cảnh phát triển theo lượt. |
| Chiếu | "Chế độ xem có phạm vi" | Chế độ xem theo vai trò cụ thể vào nhóm được chia sẻ để tránh cồng kềnh ngữ cảnh. |

## Đọc thêm

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — triển khai tham chiếu
- [AG2 repo](https://github.com/ag2ai/ag2) — cộng đồng AutoGen v0.2 tiếp tục
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) - người kế nhiệm merged, RC Tháng Hai 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — diễn viên theo hướng sự kiện model viết lại chi tiết
