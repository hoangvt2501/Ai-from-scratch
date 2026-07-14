# Chuyên môn hóa vai trò - Người lập kế hoạch, Nhà phê bình, Người thực hiện, Người xác minh

> Sự phân rã đa agent phổ biến nhất vào năm 2026: một agent lập kế hoạch, một người thực hiện, một người phê bình hoặc xác minh. MetaGPT (arXiv:2308.00352) chính thức hóa điều này dưới dạng SOP được mã hóa thành vai trò prompts - Giám đốc sản phẩm, Kiến trúc sư, Giám đốc dự án, Kỹ sư, Kỹ sư QA - sau `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) chuỗi nhà thiết kế, lập trình viên, người đánh giá, người kiểm tra thông qua một "chuỗi trò chuyện" với "ảo giác giao tiếp" (agents yêu cầu rõ ràng các chi tiết còn thiếu). Trình xác minh chịu tải: Cemri et al. (MAST, arXiv: 2503.13657) cho thấy mọi lỗi nhiều agent đều có thể được truy nguyên để xác minh bị thiếu hoặc bị hỏng. PwC báo cáo mức tăng 7× accuracy (10% → 70%) từ các vòng xác thực có cấu trúc trong CrewAI.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model), Giai đoạn 16 · 05 (Giám sát)
**Thời lượng:** ~60 phút

## Vấn đề

Các hệ thống đa agent chung tạo ra đầu ra chung. Ba lập trình viên trong một cuộc trò chuyện nhóm viết ba hương vị của cùng một mã tầm thường. Bạn có thể thêm nhiều agents, thêm nhiều vòng mà vẫn không vượt qua ngưỡng chất lượng.

Bản sửa lỗi không agents hơn - nó *khác *agents. Chỉ định các vai trò riêng biệt. Cung cấp cho các công cụ phê bình mà người lập kế hoạch không có. Cung cấp cho người xác minh một bộ kiểm tra khách quan. Bây giờ hệ thống có sự bất đồng nội bộ với việc sửa chữa có cơ sở, không chỉ là phỏng đoán song song.

## Khái niệm

### Bốn vai trò kinh điển

**Planner.** Đọc mục tiêu, tạo danh sách bước hoặc thông số kỹ thuật. Công cụ: truy xuất kiến thức, tài liệu. Đầu ra: kế hoạch có cấu trúc.

**Executor.** Đọc từng bước kế hoạch, tạo ra artifact. Công cụ: các công cụ làm việc thực tế (trình biên dịch mã, shell API client). Đầu ra: artifact.

**Critic.** Đọc kết quả của người thực thi trái với ý định của người lập kế hoạch. Công cụ: chỉ đọc quyền truy cập vào artifact, phân tích tĩnh. Đầu ra: accept/reject có lý do.

**Verifier.** Đọc artifact và chạy kiểm tra xác định. Công cụ: chạy thử nghiệm, trình kiểm tra kiểu schema trình xác thực. Đầu ra: pass/fail có bằng chứng.

Nhà phê bình chủ quan, cố chấp, thường dựa trên LLM. Trình xác minh là khách quan, xác định, thường dựa trên mã. Họ không phải là cùng một vai trò.

### Mô hình SOP của MetaGPT

MetaGPT (arXiv:2308.00352) mã hóa SOP kỹ thuật phần mềm dưới dạng vai trò prompts:

- **Giám đốc sản phẩm** viết PRD.
- **Kiến trúc sư** sản xuất thiết kế hệ thống.
- **Trình quản lý dự án** phân chia nhiệm vụ.
- **Kỹ sư** thực hiện.
- **Kỹ sư QA** chạy thử nghiệm.

Mỗi vai trò đều có một input/output schema nghiêm ngặt. Vai trò prompt nói vai trò * là gì * và nó * phải tạo ra cái gì*. Công thức `Code = SOP(Team)` - SOP xác định biến một nhóm LLMs thành một pipeline có thể dự đoán được.

### Ảo giác giao tiếp của ChatDev

ChatDev bổ sung một động thái quan trọng: khi người thực thi cần một chi tiết cụ thể không có trong kế hoạch, nó sẽ hỏi rõ ràng nhà thiết kế trước khi tiếp tục. Điều này ngăn chặn sự thất bại LLM cổ điển của việc phát minh ra chi tiết một cách hợp lý.

Thực hiện: vai trò prompt bao gồm "khi bạn cần thông tin cụ thể mà bạn không được cung cấp, hãy hỏi tên vai trò liên quan trước khi đưa ra kết quả".

### Tại sao trình xác minh lại quan trọng nhất

Cemri et al. (MAST) đã theo dõi 1642 lỗi thực thi nhiều agent. 21,3% là lỗ hổng xác minh - hệ thống shipped câu trả lời chưa ai kiểm tra. 79% còn lại thường trace trở lại "có một cuộc kiểm tra không thành công hoặc không bao giờ được chạy". Xác minh là vai trò chịu tải.

PwC đã báo cáo (Triển khai CrewAI, 2025) rằng việc thêm vòng lặp xác thực có cấu trúc đã chuyển accuracy từ 10% lên 70%. 7× lợi nhuận từ một vai trò.

### Nhà phê bình vs người xác minh

- Một nhà phê bình là một LLM đánh giá một artifact về chất lượng. Chủ quan. Có thể bị lừa bởi văn xuôi hợp lý.
- Trình xác minh là một chương trình xác định chạy trên artifact. Mục tiêu. Cung cấp cho pass/fail bằng chứng.

Sử dụng cả hai. Nhà phê bình nắm bắt các vấn đề về thị hiếu mà người xác minh không thể nói rõ. Verifier bắt các lỗi mà nhà phê bình không thể nhìn thấy vì chúng chỉ xuất hiện ở runtime.

### Chống mô hình

Mỗi vai trò trong hệ thống của bạn là một LLM và đầu ra của mọi vai trò đều "có vẻ tốt đối với tôi". Chế độ lỗi MAST cổ điển. Thêm ít nhất một trình xác minh có pass/fail được quyết định bằng mã chứ không phải bằng LLM.

### Ánh xạ Framework

- **CrewAI** — `Agent(role, goal, backstory)` là bề mặt chuyên môn hóa sách giáo khoa.
- **LangGraph** — các nút có thể có prompts chuyên biệt; các cạnh thực thi pipeline.
- **AutoGen** — ConversableAgents theo vai trò cụ thể với tên một từ trong GroupChat.
- **OpenAI Agents SDK** — công cụ bàn giao giữa các Agents chuyên môn về vai trò.

## Tự xây dựng

`code/main.py` triển khai 4 vai trò pipeline xây dựng một hàm Python đơn giản:

- **Planner** tạo ra một thông số kỹ thuật.
- **Executor** tạo một chuỗi mã.
- **Nhà phê bình **(mô phỏng LLM) đánh dấu các vấn đề rõ ràng.
- **Verifier** chạy mã được tạo trong một sandbox (`exec`) đối với một trường hợp thử nghiệm.

Demo chạy hai lần: một lần khi trình thực thi tạo ra mã chính xác (cả hai trình phê bình + trình xác minh đều đạt), một lần khi trình thực thi tạo ra mã ngoài thông số kỹ thuật (người phê bình bỏ lỡ lỗi vì nó có vẻ hợp lý, trình xác minh bắt nó vì kiểm tra thất bại).

Chạy:

```
python3 code/main.py
```

## Ứng dụng

`outputs/skill-role-designer.md` nhận một nhiệm vụ và tạo ra danh sách vai trò (3-5 vai trò), input/output schema cho mỗi vai trò và kiểm tra người xác minh. Sử dụng điều này trước khi nối dây agents vào framework.

## Sản phẩm bàn giao

Danh sách kiểm tra:

- **Ít nhất một người xác minh xác định.** Không bao giờ LLM tất cả.
- **I/O schema rõ ràng cho mỗi vai trò.** Người lập kế hoạch trả về một thông số kỹ thuật, không phải văn xuôi; Người thi hành đọc schema đó.
- **Ảo giác giao tiếp.** Người thực thi phải hỏi người lập kế hoạch khi thiếu thông tin; không bao giờ phát minh ra nó.
- **Critic/verifier đặt hàng.** Chạy nhà phê bình trước (rẻ, bắt được các vấn đề thiết kế), trình xác minh thứ hai (chậm, bắt lỗi).
- **Vòng lặp ngân sách.** Tối đa 2 vòng sửa đổi phê bình-thực thi trước khi leo thang lên con người.

## Bài tập

1. Chạy `code/main.py` và quan sát cách trình xác minh phát hiện lỗi mà nhà phê bình đã bỏ lỡ. Thêm kiểm tra phân tích tĩnh (đếm số lần xuất hiện `return`) làm trình xác minh bổ sung. Bài kiểm tra runtime bị bỏ lỡ điều gì?
2. Thêm vai trò thứ 5: "nhà phân tích yêu cầu" chuyển mong muốn của người dùng thành thông số kỹ thuật sẵn sàng cho người lập kế hoạch. Những yêu cầu khử ảo giác giao tiếp nào nên chảy đến nó?
3. Đọc MetaGPT Phần 3 ("Agents"). Liệt kê input/output schema của từng vai trò trong số 5 vai trò của MetaGPT.
4. Đọc sơ đồ chuỗi trò chuyện của ChatDev (arXiv: 2307.07924 Hình 3). Xác định nơi ảo giác giao tiếp phá vỡ một vòng lặp mà nếu không sẽ là vô hạn.
5. Mức tăng 7× accuracy của PwC đến từ các vòng lặp xác minh. Đưa ra giả thuyết ba nhiệm vụ mà việc thêm một trình xác minh sẽ không giúp ích gì - trong đó việc kiểm tra tính đúng đắn là không thể hoặc quá tốn kém.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Chuyên môn hóa vai trò | "Khác agents, công việc khác nhau" | Hệ thống riêng biệt prompts điều chỉnh cho các vai trò planner/executor/critic/verifier. |
| Mô hình SOP | "Quy trình vận hành tiêu chuẩn được mã hóa" | Khung của MetaGPT: I/O schemas nghiêm ngặt cho mỗi vai trò biến một đội thành một pipeline. |
| Ảo giác giao tiếp | "Hỏi trước khi phát minh" | Mẫu ChatDev: người thực thi hỏi người lập kế hoạch khi thiếu một chi tiết thay vì tạo ra một chi tiết. |
| Nhà phê bình | "Người đánh giá LLM" | Người đánh giá chủ quan, cố chấp. Nắm bắt các vấn đề về hương vị. Có thể bị lừa bởi văn xuôi hợp lý. |
| Người xác minh | "Kiểm tra xác định" | Trình chạy thử nghiệm pass/fail. dựa trên mã, trình kiểm tra loại schema trình xác thực. Không thể bị lừa. |
| Khoảng cách xác minh | "Không ai kiểm tra" | 21,3% thất bại của MAST. Trả lời shipped mà không cần kiểm tra sẽ phát hiện ra lỗi. |
| Vòng lặp sửa đổi | "Nhà phê bình gửi nó trở lại" | Từ chối phê bình triggers người thực thi chạy lại với phản hồi. Cần ngân sách. |
| Chống hoa văn toàn LLM | "Có vẻ tốt đối với tôi" | Mỗi vai trò là một LLM, không có kiểm tra xác định. Lỗi MAST cổ điển. |

## Đọc thêm

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — tài liệu tham khảo SOP-as-role-prompt
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) - chuỗi trò chuyện + ảo giác giao tiếp
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - phân loại MAT; Khoảng trống xác minh là 21,3% số lỗi
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) - bề mặt đặc điểm kỹ thuật vai trò production
