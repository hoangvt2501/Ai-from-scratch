# Skill Thư viện và Học tập suốt đời (Voyager)

> Voyager (Wang et al., TMLR 2024) coi mã thực thi như một skill. Skills được đặt tên, có thể truy xuất, có thể kết hợp và tinh chỉnh theo phản hồi của môi trường. Đây là kiến trúc tham chiếu cho Claude Agent SDK skills, skillkit và mẫu thư viện skill 2026.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 07 (MemGPT), Giai đoạn 14 · 08 (Khối Letta)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Kể tên ba thành phần của Voyager - chương trình giảng dạy tự động, thư viện skill, prompting lặp đi lặp lại - và vai trò của từng thành phần.
- Giải thích lý do tại sao Voyager tạo mã không gian hành động, không phải lệnh primitive.
- Triển khai thư viện skill stdlib với tính năng đăng ký, truy xuất, thành phần và tinh chỉnh do lỗi.
- Ánh xạ mô hình của Voyager vào Claude Agent SDK skills 2026 và hệ sinh thái bộ kỹ năng.

## Vấn đề

Agents xây dựng lại mọi khả năng từ đầu trong mỗi session làm sai ba điều:

1. **Lãng phí tokens.** Mọi nhiệm vụ đều gợi ra cùng một lý luận.
2. **Mất tiến độ.** Một hiệu chỉnh đã học được trong session A không chuyển sang session B.
3. **Thất bại trong thành phần dài hạn.** Các tác vụ phức tạp cần có hệ thống phân cấp khả năng; one-shot prompts không thể diễn tả chúng.

Câu trả lời của Voyager: coi mỗi khả năng có thể tái sử dụng như một đoạn mã được đặt tên được lưu trữ trong thư viện, có thể truy xuất bằng sự tương tự, có thể kết hợp với các skills khác và được tinh chỉnh bằng phản hồi thực thi.

## Khái niệm

### Ba thành phần

Voyager (arXiv:2305.16291) cấu trúc một agent xung quanh:

1. **Chương trình giảng dạy tự động.** Người đề xuất theo sự tò mò sẽ chọn nhiệm vụ tiếp theo dựa trên trạng thái môi trường và skill hiện tại của agent. Khám phá là từ dưới lên.
2. **Skill thư viện.** Mỗi skill là mã thực thi. Các skills mới được thêm vào khi một nhiệm vụ thành công. Skills được truy xuất bằng sự tương đồng từ truy vấn đến mô tả.
3. **Cơ chế prompting lặp lại.** Khi bị lỗi, agent nhận lỗi thực thi, phản hồi môi trường và đầu ra tự xác minh, sau đó tinh chỉnh skill.

Đánh giá Minecraft (Wang và cộng sự, 2024): nhiều vật phẩm độc đáo hơn 3,3 lần, công cụ đá nhanh hơn 8,5 lần, công cụ sắt nhanh hơn 6,4 lần, di chuyển bản đồ dài hơn 2,3 lần so với đường cơ sở. Các con số dành riêng cho Minecraft, nhưng mẫu chuyển đổi.

### Không gian hành động = mã

Hầu hết agents phát ra các lệnh primitive. Voyager phát ra JavaScript chức năng. Một skill là:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Sáng tác từ sub-skills. Được lưu trữ theo mô tả và embedding. Được truy xuất dưới dạng một chương trình, không phải prompt.

Đây là Claude Agent SDK skill 2026: một đoạn mã được đặt tên, có thể truy xuất cộng với các hướng dẫn mà agent tải theo yêu cầu.

### Truy xuất Skill

Nhiệm vụ mới "làm cuốc kim cương". Agent:

1. Nhúng mô tả nhiệm vụ.
2. Truy vấn thư viện skill để tìm skills tương tự top-k.
3. Truy xuất `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable`, v.v.
4. Soạn skill mới từ primitives được truy xuất + logic mới.

Đây là mẫu MCP tài nguyên (Giai đoạn 13) và Agent SDK skills thực hiện: truy xuất trên bề mặt knowledge/code, phạm vi nhiệm vụ hiện tại.

### Tinh chỉnh lặp đi lặp lại

Vòng phản hồi của Voyager:

1. Agent viết một skill.
2. Skill chạy ngược lại với môi trường.
3. Một trong ba tín hiệu trả về: `success`, `error` (với stack trace), `self-verification failure`.
4. Agent viết lại skill bằng cách sử dụng tín hiệu làm ngữ cảnh.
5. Lặp lại cho đến khi thành công hoặc vòng tối đa.

Đây là Tự tinh chỉnh (Bài 05) áp dụng cho việc tạo mã với xác minh dựa trên môi trường. CRITIC (Bài 05) là mẫu tương tự với các công cụ bên ngoài với trình xác minh.

### Chương trình giảng dạy và khám phá

Mô-đun chương trình giảng dạy của Voyager đề xuất các nhiệm vụ như "xây dựng một nơi trú ẩn gần hồ" dựa trên những gì agent có và những gì nó chưa làm. Người đề xuất sử dụng trạng thái môi trường + khoảng không quảng cáo skill để chọn một nhiệm vụ ngay trên khả năng hiện tại - điểm ngọt ngào khám phá.

Đối với production agents, điều này có nghĩa là toán tử "những gì còn thiếu": với thư viện skill hiện tại và một tên miền, chúng ta chưa đề cập đến những skills gì? Các nhóm thường triển khai điều này theo cách thủ công dưới dạng đánh giá chương trình giảng dạy.

### Mô hình này sai ở đâu

- **Skill thối thư viện.** Cùng một skill được thêm 10 lần với các mô tả hơi khác nhau. Thêm tính năng khử trùng lặp khi ghi; Truy xuất chỉ trả về một.
- **Trôi dạt skill sáng tác.** Parent skill phụ thuộc vào một đứa trẻ đã được tinh luyện. Phiên bản skills; Cha mẹ được ghim vào v1 không chọn v3 một cách kỳ diệu.
- **Chất lượng truy xuất.** Vector truy xuất trên skill mô tả sẽ giảm khi thư viện phát triển vượt qua vài trăm. Bổ sung với bộ lọc thẻ và các ràng buộc cứng ("chỉ skills với `category=tooling`").

## Tự xây dựng

`code/main.py` triển khai thư viện skill stdlib:

- `Skill` — tên, mô tả, mã (dưới dạng chuỗi), phiên bản, thẻ, phần phụ thuộc.
- `SkillLibrary` - Đăng ký, tìm kiếm (token trùng lặp), Compose (loại cấu trúc liên kết của deps) và tinh chỉnh (tăng phiên bản khi cập nhật).
- Một agent có kịch bản đăng ký ba primitive skills, sáng tác thứ tư, thất bại và tinh chỉnh.

Chạy nó:

```
python3 code/main.py
```

trace hiển thị các lần ghi, truy xuất, kết hợp, thực thi không thành công và tinh chỉnh v2 - vòng lặp của Voyager từ đầu đến cuối.

## Ứng dụng

- **Claude Agent SDK skills** (Anthropic) — Tài liệu tham khảo năm 2026: mỗi skill có một mô tả, mã và hướng dẫn; được tải theo yêu cầu trong một agent session.
- **SkillKit** (npm: Skillkit) — Quản lý agent skill chéo cho 32+ AI agents mã hóa.
- **Thư viện skill tùy chỉnh** — miền cụ thể (SQL skills cho agents dữ liệu, Terraform skills cho agents hạ tầng). Mô hình Voyager thu nhỏ lại.
- **OpenAI Agents SDK `tools`** - ở mức thấp; Mỗi công cụ là một skill nhẹ.

## Sản phẩm bàn giao

`outputs/skill-skill-library.md` tạo thư viện skill hình Voyager với đăng ký, truy xuất, lập phiên bản và tinh chỉnh được kết nối cho bất kỳ runtime đích nào.

## Bài tập

1. Thêm trình phát hiện chu kỳ phụ thuộc vào `compose()`. Điều gì xảy ra khi skill A phụ thuộc vào B phụ thuộc vào A? Lỗi vs cảnh báo?
2. Triển khai tính năng ghim phiên bản trên mỗi skill. Khi cha mẹ skill soạn `crafting@1` con, việc tinh chỉnh `crafting@2` không được âm thầm nâng cấp cha mẹ.
3. Thay thế truy xuất chồng chéo token bằng câu transformers embeddings (hoặc BM25 stdlib impl). Đo retrieval@5 trên thư viện đồ chơi 50 skill.
4. Thêm một agent "chương trình giảng dạy": với thư viện hiện tại và mô tả tên miền, đề xuất 5 skills còn thiếu. Gọi nó là hàng tuần.
5. Đọc tài liệu Claude Agent SDK skill của Anthropic. Chuyển thư viện đồ chơi vào skill schema của SDK. Những thay đổi nào về khả năng khám phá?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Skill | "Khả năng tái sử dụng" | Đoạn mã + mô tả được đặt tên, có thể truy xuất theo sự tương đồng |
| Thư viện Skill | "Agent ký ức về cách thực hiện" | Kho lưu trữ liên tục của skills, có thể tìm kiếm và có thể kết hợp |
| Chương trình giảng dạy | "Người đề xuất nhiệm vụ" | Trình tạo mục tiêu từ dưới lên được thúc đẩy bởi khoảng cách năng lực hiện tại |
| Thành phần | "Skill DAG" | Skills viện dẫn skills; sắp xếp tô pô khi thực hiện |
| Tinh chỉnh lặp đi lặp lại | "Vòng lặp tự sửa" | Phản hồi Env + lỗi + tự xác minh quay trở lại phiên bản tiếp theo |
| Không gian hành động dưới dạng mã | "Hành động có lập trình" | Phát ra các chức năng, không phải lệnh primitive, cho hành vi mở rộng theo thời gian |
| Dedup khi ghi | "Skill sụp đổ" | Các mô tả gần như trùng lặp thu gọn thành một skill chính tắc |

## Đọc thêm

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — bài báo skill thư viện ban đầu
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - skills là sản phẩm hóa năm 2026
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - skills và subagents trong thực tế
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — vòng lặp tinh chỉnh bên dưới Voyager
