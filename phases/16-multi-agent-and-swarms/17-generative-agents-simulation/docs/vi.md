# Agents tổng quát và mô phỏng nổi lên

> Park et al. 2023 (UIST '23, arXiv:2304.03442) đã điền vào **Smallville**, một sandbox gồm 25 agents, với kiến trúc gồm ba phần: **luồng bộ nhớ** (nhật ký ngôn ngữ tự nhiên), **phản chiếu** (tổng hợp cấp cao hơn mà agent tạo ra về luồng của chính nó) và **plan** (hành vi cấp ngày, sau đó là kế hoạch phụ). Kết quả mang tính bước ngoặt là sự xuất hiện của bữa tiệc Ngày lễ tình nhân: một agent được gieo hạt giống với "muốn tổ chức một bữa tiệc Ngày lễ tình nhân", mà không cần viết thêm kịch bản, tạo ra những lời mời lan truyền khắp dân chúng, ngày tháng được phối hợp và bữa tiệc đã diễn ra - từ 24 agents những người bắt đầu mà không biết gì về nó. Cắt bỏ cho thấy cả ba thành phần đều cần thiết cho độ tin cậy. Các lỗi được ghi nhận là lỗi tiêu chuẩn không gian (vào các cửa hàng đóng cửa, dùng chung phòng tắm cho một người). Đây là kiến trúc tham chiếu cho các mô phỏng agent và đánh giá xã hội đa agent vào năm 2026.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 04 (Primitive Model), Giai đoạn 16 · 13 (Bộ nhớ chia sẻ)
**Thời lượng:** ~75 phút

## Vấn đề

Hầu hết các hệ thống đa agent là các nhóm có kịch bản chặt chẽ: kế hoạch lập kế hoạch, mã lập trình, đánh giá của người đánh giá. Điều đó hoạt động cho các nhiệm vụ được xác định rõ ràng. Nó không nắm bắt được hành vi mới nổi, không có kịch bản phát sinh khi agents có trí nhớ, ưu tiên và một thế giới mở. Nghiên cứu, mô phỏng xã hội và ngày càng có nhiều AI trò chơi cần loại thứ hai này.

Kiến trúc Smallville là benchmark cho nó. Cho đến Park 2023, mô phỏng agent tốt nhất là những người theo dõi script nông cạn; Sau đó, mô hình là mặc định cho agents tạo sinh trong thế giới mở. Nếu bạn xây dựng một mô phỏng agent vào năm 2026, bạn đang sử dụng ba thành phần của Smallville hoặc biện minh rõ ràng tại sao bạn không sử dụng.

## Khái niệm

### Ba thành phần

**Luồng bộ nhớ.** Nhật ký chỉ bổ sung các quan sát, hành động, phản ánh và kế hoạch. Mỗi mục nhập có dấu thời gian, loại, mô tả (ngôn ngữ tự nhiên) và siêu dữ liệu dẫn xuất: **gần đây**, **tầm quan trọng** (tự xếp hạng 1-10 theo agent) và **liên quan** (tương tự cosin với truy vấn hiện tại).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Truy xuất bộ nhớ kết hợp ba điểm số: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Top-k mục nhập vào prompt hiện tại.

**Phản ánh.** Định kỳ (mỗi N ký ức hoặc về các sự kiện quan trọng), agent tạo ra các tổng hợp bậc cao hơn từ những ký ức gần đây. Các mục phản xạ quay trở lại luồng và có thể truy xuất được như bất kỳ bộ nhớ nào khác. Đây là cách agents xây dựng "sự hiểu biết" - tương đương với niềm tin lâu dài của kiến trúc.

**Kế hoạch.** Phân hủy từ trên xuống. Đầu tiên, một kế hoạch cấp ngày theo nét rộng ("đi làm, ăn tối với Klaus"). Sau đó là các gói cấp giờ. Sau đó là các kế hoạch cấp hành động. Kế hoạch có thể sửa đổi: khi một quan sát mâu thuẫn với một kế hoạch, agent lập kế hoạch lại phân đoạn bị ảnh hưởng.

### Tại sao cả ba đều quan trọng (cắt bỏ)

Park và cộng sự đã thực hiện cắt bỏ từng quan sát, phản ánh và lập kế hoạch. Mỗi lần cắt bỏ đều làm tổn hại đến độ tin cậy:

- Nếu không có **quan sát**, agent sẽ bỏ lỡ bối cảnh và hành động theo những niềm tin cũ kỹ.
- Nếu không có **phản ánh**, agent không thể hình thành niềm tin bậc cao; tương tác vẫn nông cạn.
- Nếu không có **kế hoạch**, hành vi trở thành nhiễu phản ứng; mục tiêu tan biến.

Điểm đáng tin cậy từ những người đánh giá là cao nhất với cả ba; Bỏ bất kỳ một cái nào tạo ra một hồi quy có thể đo lường được.

### Sự xuất hiện của ngày lễ tình nhân

Một agent, Isabella Rodriguez, được xếp hạt giống với mục tiêu "muốn tổ chức một bữa tiệc Ngày lễ tình nhân tại Hobbs Cafe vào lúc 5 giờ chiều ngày 14 tháng 2". 24 agents còn lại không nhận được hạt giống như vậy. Trong những ngày mô phỏng:

1. Kế hoạch của Isabella bao gồm việc mời mọi người.
2. Mỗi lời mời trở thành một quan sát trong luồng ký ức của hàng xóm.
3. Sự phản ánh của người hàng xóm đó tạo ra niềm tin: "Isabella đang tổ chức một bữa tiệc."
4. Kế hoạch của người hàng xóm bao gồm "tham dự bữa tiệc vào ngày 14 tháng 2".
5. Hàng xóm nói với những người hàng xóm khác. Lời mời lan truyền mà không có sự phối hợp trung tâm.
6. Vào lúc 5 giờ chiều ngày 14 tháng 2, một số agents hội tụ tại Hobbs Cafe.

Đây là sự xuất hiện theo nghĩa kỹ thuật: hành vi cấp hệ thống (một bên) phát sinh từ các tương tác cục bộ (lời mời song phương + lập kế hoạch cá nhân) mà không có người điều phối trung tâm.

### Các chế độ lỗi được ghi lại

Park và cộng sự tài liệu rõ ràng:

- **Lỗi định mức không gian.** Agents bước vào các cửa hàng đóng cửa. Agents cố gắng sử dụng cùng một phòng tắm dành cho một người. Agents ăn trong phòng không dành cho ăn. model không suy ra các chuẩn mực vật lý xã hội chỉ từ môi trường.
- **Tràn bộ nhớ.** Chạy mô phỏng sâu khiến chi phí truy xuất bộ nhớ tăng lên. Biện pháp khắc phục thực tế: nén bộ nhớ định kỳ (tóm tắt và cắt tỉa) và phân rã trên các mục có tầm quan trọng thấp.
- **Ảo giác phản chiếu.** Phản xạ có thể tạo ra các mối quan hệ không tồn tại trong luồng ký ức. Giảm thiểu: bao gồm id bộ nhớ nguồn trong prompts phản xạ và xác minh tại thời điểm truy xuất.

Đây là những chế độ lỗi liên quan đến production: bất kỳ mô phỏng agent 2026 nào cũng kế thừa chúng.

### Quy tắc triển khai ba thành phần

1. **Bộ nhớ chỉ được thêm vào.** Không bao giờ thay đổi mục nhập bộ nhớ. Sửa chữa là các mục mới.
2. **Điểm quan trọng rẻ.** Gọi cho LLM để đánh giá mức độ quan trọng từ 1-10 tại thời điểm viết. Lưu điểm số vào bộ nhớ đệm.
3. **Truy xuất được xếp hạng, không được lọc.** Top-k theo điểm tổng hợp; Không sử dụng bộ lọc cứng (làm mất ngữ cảnh).
4. **Phản xạ chạy định kỳ.** Trigger khi tổng tầm quan trọng của các ký ức chưa được xử lý vượt quá ngưỡng (ví dụ: 150).
5. **Kế hoạch có thể sửa đổi.** Khi một quan sát mới mâu thuẫn với một kế hoạch, chỉ tạo lại phân đoạn bị ảnh hưởng, không phải toàn bộ kế hoạch.

### Generative agents ngoài Smallville

Tài liệu tiếp theo 2024-2026 mở rộng kiến trúc:

- **Mô phỏng xã hội đa agent để nghiên cứu policy / thị trường.** Các quần thể giống như Smallville mô phỏng hành vi của người dùng để phản ứng với features. Nhanh hơn A/B xét nghiệm; accuracy đang tranh chấp.
- **NPC AI cho trò chơi.** Game nhập vai với Smallville agents tạo ra cốt truyện mới nổi thay vì các nhiệm vụ theo kịch bản.
- **Đánh giá agent tổng quát benchmarks.** Thay vì accuracy nhiệm vụ, số liệu trở thành độ tin cậy + sự gắn kết của hành vi trong thời gian dài.

Kiến trúc là tài liệu tham khảo. Các phần mở rộng hoán đổi các thành phần (vector lưu trữ bộ nhớ, phản xạ tăng cường truy xuất, kế hoạch biểu tượng thần kinh) nhưng giữ cấu trúc ba phần.

### Tại sao điều này lại quan trọng đối với kỹ thuật đa agent

Smallville là bằng chứng về khái niệm rằng sự xuất hiện đa agent là rẻ khi các thành phần đúng. Kiến trúc hiện đã được sao chép trên models mã nguồn mở (nhỏ hơn LLMs mất độ tin cậy một cách duyên dáng, không mạnh). Bất kỳ hệ thống production nào cần **hành vi xã hội mới nổi** đều sử dụng hình dạng này. Bất kỳ hệ thống nào cần **thực hiện nhiệm vụ chặt chẽ** đều sử dụng các mẫu giám sát / vai trò / primitives từ trước đó trong giai đoạn này.

## Tự xây dựng

`code/main.py` triển khai ba thành phần trong stdlib Python với các agent policies kịch bản (không có LLM thực). Bản demo tái tạo sự xuất hiện của bữa tiệc Valentine trong thu nhỏ:

- `MemoryStream` — nhật ký chỉ nối với recency/importance/relevance truy xuất.
- `reflect(stream)` - suy ngẫm theo kịch bản về những ký ức quan trọng gần đây.
- `plan(agent_state)` - kế hoạch cấp ngày và cấp giờ dựa trên niềm tin hiện tại.
- Tình huống: 5 agents. Agent 1 bắt đầu với "tổ chức tiệc lúc 5 giờ chiều". Qua các tick mô phỏng, lời mời lan rộng và agents hội tụ.

Chạy:

```
python3 code/main.py
```

Đầu ra dự kiến: trace từng tích tắc. Đến tích cuối cùng, ít nhất 3 trong số 5 agents thể hiện bữa tiệc trong kế hoạch của họ và họ hội tụ tại địa điểm tổ chức bữa tiệc. Hạt giống duy nhất tạo ra sự xuất hiện phối hợp mà không có bất kỳ người điều phối nào.

## Ứng dụng

`outputs/skill-simulation-designer.md` thiết kế mô phỏng agent tổng hợp: số agents, schema bộ nhớ, nhịp phản chiếu, chân trời kế hoạch và số liệu đánh giá.

## Sản phẩm bàn giao

Quy tắc mô phỏng production:

- **Bộ nhớ là cơ sở dữ liệu.** Chọn một kho lưu trữ thực (vector DB, Postgres) trên quy mô lớn. Stdlib trong bộ nhớ dành cho nguyên mẫu.
- **Ghi lại trace truy xuất.** Đối với mọi hành động, hãy ghi lại top-k ký ức đã thúc đẩy hành động đó. Đây là khả năng gỡ lỗi của bạn.
- **Ngân sách trên mỗi agent tokens.** Truy xuất + phản ánh + gói trên mỗi tick của mỗi agent là O(k) LLM cuộc gọi. Đánh dấu N agents × T × cuộc gọi mỗi lần đánh dấu có thể làm giảm ngân sách của bạn.
- **Bộ nhớ nhỏ gọn định kỳ.** Tóm tắt và cắt bớt các mục có tầm quan trọng thấp. Lưu giữ policy là một quyết định thiết kế, không phải là một chi tiết.
- **Phát hiện rõ ràng các vi phạm chuẩn mực không gian/xã hội**. Kiến trúc không học chúng.

## Bài tập

1. Chạy `code/main.py`. Xác nhận 3+ agents hội tụ tại bữa tiệc. Tăng agents lên 10 - sự xuất hiện có còn xảy ra không?
2. Loại bỏ bước phản chiếu. Hành vi trông như thế nào? Bản đồ đến kết quả cắt bỏ ở Park 2023.
3. Giới thiệu một mục tiêu hạt giống cạnh tranh ("Klaus muốn nói chuyện nghiên cứu lúc 5 giờ chiều"). agents chia tay, hay một mục tiêu chiếm ưu thế? Điều gì quyết định nó?
4. Thêm các ràng buộc về không gian: Hobbs Cafe chứa tối đa 4 agents. Mô phỏng có xử lý tràn một cách duyên dáng hay nó đánh vào mô hình lỗi "phòng tắm một người"?
5. Đọc Park et al. (arXiv: 2304.03442) Phần 6 (thí nghiệm hành vi mới nổi). Xác định một hành vi không thể tái tạo trong thu nhỏ của bạn. Bạn cần cải thiện thành phần nào của kiến trúc?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Luồng bộ nhớ | "Nhật ký của agent" | Chỉ thêm nhật ký quan sát, hành động, phản ánh, kế hoạch. |
| Gần đây | "Ký ức mới làm sao" | Điểm phân rã theo cấp số nhân theo tuổi. |
| Tầm quan trọng | "Người agent quan tâm đến mức nào" | Tự đánh giá 1-10 tại thời điểm viết. Bộ nhớ đệm. |
| Mức độ liên quan | "Liên quan đến truy vấn hiện tại như thế nào" | Sự tương đồng cosin (dựa trên embedding). |
| Suy ngẫm | "Niềm tin bậc cao" | Tổng hợp được tạo ra từ những ký ức gần đây, được nhập lại như một ký ức mới. |
| Kế hoạch | "Day/hour/action phân hủy" | Cây kế hoạch từ trên xuống. Có thể sửa đổi khi các quan sát mâu thuẫn. |
| Smallville | "Công viên 2023 sandbox" | Mô phỏng 25 agent đã tạo ra sự xuất hiện của Ngày lễ tình nhân. |
| Độ tin cậy | "Chỉ số chất lượng" | Điểm đánh giá của con người xem liệu hành vi có giống như một agent hợp lý hay không. |

## Đọc thêm

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — kiến trúc tham chiếu
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — địa điểm xuất bản
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — tham khảo Python triển khai
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) - prior nghệ thuật cho agents trí nhớ có cấu trúc
