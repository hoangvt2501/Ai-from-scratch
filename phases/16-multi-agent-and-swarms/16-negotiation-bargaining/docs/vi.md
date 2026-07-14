# Đàm phán và thương lượng

> Agents thương lượng nguồn lực, giá cả, phân bổ nhiệm vụ và điều khoản. Bộ benchmark năm 2026 rất rõ ràng: NegotiationArena (arXiv:2402.05863) cho thấy LLMs có thể cải thiện lợi nhuận ~20% thông qua thao túng tính cách ("tuyệt vọng"); "Đo lường khả năng thương lượng" (arXiv: 2402.15813) cho thấy người mua khó hơn người bán và quy mô không giúp ích được gì - **OG-Narrator** (trình tạo đề nghị xác định + LLM người kể chuyện) của họ đã đẩy tỷ lệ giao dịch từ 26,67% lên 88,88%; Cuộc thi đàm phán tự động quy mô lớn (arXiv: 2503.06416) đã thực hiện ~180k cuộc đàm phán và nhận thấy rằng **chain-of-thought che giấu **agents giành chiến thắng bằng cách che giấu lý do với các đối tác; Bhattacharya et al. 2025 về các chỉ số của Dự án Đàm phán Harvard xếp hạng Llama-3 hiệu quả nhất, Claude-3 tích cực GPT-4 công bằng nhất. Bài học này triển khai Contract Net Protocol (tổ tiên của FIPA, Bài 02), kết nối một buyer/seller kiểu LLM, chạy phân tách kiểu OG-Narrator và đo lường tỷ lệ giao dịch thay đổi như thế nào với mỗi lựa chọn cấu trúc.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 16 · 02 (Di sản FIPA-ACL), Giai đoạn 16 · 09 (Mạng Swarm song song)
**Thời lượng:** ~75 phút

## Vấn đề

Hai agents cần phải đồng ý về một mức giá. Để lại cho riêng mình prompts ngôn ngữ thuần túy, 2024-2026 LLMs chốt giao dịch với tỷ lệ thấp đáng ngạc nhiên (~27% đối với các món hời được tham số chặt chẽ trong arXiv: 2402.15813). Quy mô không khắc phục được nó: GPT-4 không có cấu trúc thương lượng tốt hơn GPT-3.5; nó tốt hơn trong * ngôn ngữ * của thương lượng.

Vấn đề gốc rễ là LLMs kết hợp hai công việc - quyết định lời đề nghị và tường thuật lời đề nghị. OG-Narrator đã tách những điều này: một trình tạo đề nghị xác định tính toán các bước di chuyển số; LLM chỉ tường thuật. Tỷ lệ giao dịch tăng lên ~89%.

Điều này phản ánh một phát hiện đa agent cổ điển: tách cơ chế khỏi lớp giao tiếp sẽ thắng. Giao thức mạng hợp đồng (FIPA, 1996; Smith, 1980) là cơ chế thị trường nhiệm vụ tham chiếu. Cắm một LLM vào vị trí tường thuật và bạn sẽ có được một thị trường tác vụ hiện đại được hỗ trợ bởi LLM.

## Khái niệm

### Mạng lưới hợp đồng, trong một đoạn

Giao thức mạng hợp đồng năm 1980 của Smith: một **người quản lý** phát sóng **lời kêu gọi đề xuất (cfp)**; **nhà thầu **trả lời bằng **đề xuất **tin nhắn có chứa đề nghị của họ; Người quản lý chọn người chiến thắng và gửi **chấp nhận-đề xuất** cho người chiến thắng và **từ chối-đề xuất** cho người thua cuộc. Người chiến thắng thực hiện công việc. Thông báo tùy chọn: **từ chối** (nhà thầu từ chối đề xuất). FIPA đã hệ thống hóa điều này như `fipa-contract-net` giao thức tương tác.

### Tại sao OG-Narrator chiến thắng

"Đo lường khả năng thương lượng của ngôn ngữ Models" (arXiv: 2402.15813) quan sát thấy rằng:

- LLMs thường xuyên phá vỡ các quy tắc thương lượng (chào bán với giá vô nghĩa, bỏ qua ZOPA của bên kia).
- Họ neo kém (chấp nhận những lời đề nghị đầu tiên tồi; đề nghị ngược lại với số tiền tượng trưng hơn là chiến lược).
- Chỉ quy mô không khắc phục được những điều này. Các models lớn hơn tạo ra ngôn ngữ hợp lý hơn với sai lầm chiến lược tương tự.

Sự phân hủy OG-Narrator:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

Trình tạo đề nghị là một chiến lược đàm phán cổ điển: model thương lượng Rubinstein, chiến lược Zeuthen hoặc ăn miếng trả miếng đơn giản về giá. Người LLM thuật lại. Thông báo chứa giá xác định và khung ngôn ngữ tự nhiên.

Tỷ lệ giao dịch tăng vì:
- Giá cả vẫn nằm trong khu vực mặc cả.
- Mỏ neo mang tính chiến lược, không phải cảm xúc.
- LLM làm những gì nó giỏi: viết.

### Phát hiện của NegotiationArena

arXiv:2402.05863 cung cấp benchmark chuẩn. Phát hiện tiêu đề:

- LLMs có thể cải thiện lợi nhuận ~20% bằng cách áp dụng tính cách ("Tôi tuyệt vọng muốn bán nó vào thứ Sáu") - thao túng tính cách là một chiến thuật thực sự.
- Fair/cooperative agents bị lợi dụng bởi đối thủ; Phòng thủ đòi hỏi tư thế phản công rõ ràng.
- Các cặp đối xứng hội tụ đến kết quả không công bằng trên khoảng 40% các kịch bản benchmark.

Đây không phải là "LLMs là những nhà đàm phán tồi". Đó là "LLMs đàm phán quá giống con người, bao gồm cả những phần có thể khai thác".

### Chain-of-thought che giấu

Cuộc thi đàm phán tự động quy mô lớn (arXiv: 2503.06416) đã thực hiện ~180k cuộc đàm phán trên nhiều chiến lược LLM. Những người chiến thắng đã che giấu lý do của họ với các đối tác:

- Nếu một agent in "Tôi sẽ chỉ đi $75; my reservation price is $70" vào một bảng nháp hiển thị công khai, đối thủ sẽ đọc nó.
- Người chiến thắng tính toán chiến lược một cách riêng tư; Kênh đầu ra chỉ chứa ưu đãi và tường thuật yêu cầu tối thiểu.

Đây là một tiếng vang năm 2026 của lý thuyết trò chơi cổ điển (Aumann 1976 về tính hợp lý và thông tin): tiết lộ chi phí định giá cá nhân của bạn. LLMs không trực giác điều này và vui vẻ gõ những dè dặt của họ vào những traces lý luận mà đối tác có thể nhìn thấy.

Bài học kỹ thuật: tách ngữ cảnh private-scratchpad khỏi ngữ cảnh tin nhắn công khai. Không tùy chọn.

### Bhattacharya et al. 2025 — Bảng xếp hạng model

Về các chỉ số của Dự án Đàm phán Harvard (đàm phán có nguyên tắc, tôn trọng BATNA, có đi có lại về lợi ích):

- **Llama-3 **hiệu quả nhất trong việc đạt được những món hời (tỷ lệ giao dịch + tiền thưởng).
- **Claude-3** là nhà đàm phán tích cực nhất (mỏ neo cao, nhượng bộ muộn).
- **GPT-4** là công bằng nhất (variance nhỏ nhất về phần thưởng giữa các cặp).

Đây là ảnh chụp nhanh năm 2025. Vấn đề không phải là model nào thắng vào tháng 4 năm 2026 - mà là các models cơ sở khác nhau có phong cách đàm phán dai dẳng. Các quần thể không đồng nhất (Bài 15) bao gồm điều này như một nguồn đa dạng.

### Phân bổ nhiệm vụ qua Contract Net + LLM

Việc tái sử dụng hiện đại Contract Net cho LLM đa agent:

1. Manager agent phân tách một nhiệm vụ thành các đơn vị.
2. Phát `cfp` với mô tả nhiệm vụ cho worker agents.
3. Mỗi worker trả về một đề nghị: `(price, eta, confidence)` giá có thể là tokens, đơn vị tính toán hoặc đô la.
4. Người quản lý chọn người chiến thắng (một hoặc nhiều, tùy thuộc vào nhiệm vụ) và giải thưởng.
5. Những workers bị từ chối có thể tự do đấu thầu cho các nhiệm vụ khác.

Điều này vượt quá 100 workers vì sự phối hợp là phát sóng và phản hồi, không phải trò chuyện đồng bộ. Được sử dụng trong production: Các mẫu orchestration của Microsoft Agent Framework, một số triển khai LangGraph.

### Đàm phán tương tác LLM-các bên liên quan

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) giới thiệu các trò chơi có thể tính điểm nhiều bên với **điểm bí mật** và **ngưỡng chấp nhận tối thiểu**. Mỗi bên liên quan có các tiện ích riêng; LLM phải suy ra chúng từ các thông điệp. Đây là sự tổng quát hóa thương lượng hai đảng thành thành lập liên minh N-đảng. Phù hợp với production thị trường nhiệm vụ với khả năng worker không đồng nhất.

### Quy tắc tường thuật so với cơ chế

Trong tất cả các benchmarks đàm phán 2024-2026, quy tắc kỹ thuật nhất quán là:

> Hãy để LLM thuật lại. Đừng để LLM tính toán đề nghị.

Nếu đề nghị cần phải là một con số (giá, ETA, số lượng), hãy tạo ra nó một cách xác định từ trạng thái đàm phán và yêu cầu LLM tạo ra khung. Nếu đề nghị cần phải là một cấu trúc đề xuất (phân tách nhiệm vụ, phân công vai trò), hãy để LLM soạn thảo nó, nhưng xác nhận nó dựa trên schema và kiểm tra ràng buộc trước khi gửi.

## Tự xây dựng

`code/main.py` thực hiện:

- `ContractNetManager`, `ContractNetTask`, `Bid` — người quản lý + nhà thầu, phát sóng CFP, thu thập đề xuất, giải thưởng.
- `og_narrator_bargain(state, rng)` - Người mua OG-Narrator: nhượng bộ theo phong cách Zeuthen xác định về phía điểm giữa.
- `seller_response(state, rng)` — policy đề nghị đối ứng của người bán xác định (ground truth cấu trúc cho cả hai phong cách).
- `naive_llm_bargain(state, rng)` - mô phỏng một người mặc cả LLM: chọn giá có variance cao, thường nằm ngoài ZOPA.
- Đo lường: tỷ lệ giao dịch hơn 1000 bản dùng thử với giá đặt trước mới được lấy mẫu cho mỗi bản dùng thử.

Chạy:

```
python3 code/main.py
```

Sản lượng dự kiến: tỷ lệ giao dịch LLM ngây thơ ~65-75%; Tỷ lệ giao dịch OG-Narrator ~85-95%; Khoảng cách 15-25 điểm là lợi thế cấu trúc của việc phân hủy việc tạo ưu đãi từ tường thuật. Cộng với ví dụ phân bổ thị trường nhiệm vụ Contract Net với ba nhà thầu và một nhiệm vụ.

## Ứng dụng

`outputs/skill-bargainer-designer.md` thiết kế một giao thức thương lượng: ai tạo ra các đề nghị (xác định hoặc LLM), ai tường thuật, cách các bảng cào riêng tư tách biệt với tin nhắn công khai và cách giám sát tỷ lệ giao dịch.

## Sản phẩm bàn giao

Production danh sách kiểm tra thương lượng:

- **Bàn cào riêng biệt.** Trạng thái riêng tư không bao giờ đến ngữ cảnh của đối tác. Điều này là không thể thương lượng.
- **Tạo ưu đãi xác định.** Giá, số lượng, ETA: tính toán, không prompt.
- **Xác thực tất cả các ưu đãi đến** so với một schema. Từ chối các đề nghị ngoài ZOPA tại ranh giới giao thức.
- **Vòng ràng buộc.** Tối đa 3-5 vòng; Leo thang lên hòa giải viên khi bế tắc.
- **Đo lường tỷ lệ giao dịch và variance thanh toán **liên tục. Tỷ lệ giao dịch giảm là một triệu chứng - thường là một sự trôi dạt prompt hoặc một cuộc tấn công từ phía đối tác.
- **Ghi lại tất cả các đề xuất bị từ chối** với lý do xác định. Đối với các nhà quản lý Contract Net, những người thua thầu cần hiểu tại sao.

## Bài tập

1. Chạy `code/main.py`. Xác nhận OG-Narrator đánh bại những LLM ngây thơ về tỷ lệ giao dịch. Bao nhiêu?
2. Thực hiện **cải thiện lợi nhuận dựa trên tính cách** (arXiv:2402.05863) — người mua chỉ áp dụng tính cách "tuyệt vọng để mua trong tuần này" trong lời tường thuật, cung cấp trình tạo không thay đổi. Tỷ lệ giao dịch hoặc khoản thanh toán có thay đổi không?
3. Thực hiện chain-of-thought **che giấu**: duy trì một chuỗi scratchpad riêng tư không được chuyển đến đối tác. Điều gì xảy ra nếu bạn vô tình làm rò rỉ nó (mô phỏng bằng cách hoán đổi kênh)?
4. Mở rộng hợp đồng ròng cho đấu giá N-bidder với giá khởi điểm. Khi tất cả các giá thầu đều vượt quá giá dự trữ, làm thế nào để người quản lý quyết định giữa giá thấp nhất và chất lượng cao nhất? Bạn chọn quy tắc giải thưởng nào và tại sao?
5. Đọc Bhattacharya et al. 2025 về các chỉ số của Dự án Đàm phán Harvard. Thực hiện hai người mặc cả với các phong cách khác nhau (hung hăng và công bằng). Đo lường variance phần thưởng theo các cặp đối xứng và không đối xứng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Ròng hợp đồng | "Thị trường nhiệm vụ" | Smith 1980, FIPA 1996. cfp + đề xuất + accept/reject. Thị trường nhiệm vụ kinh điển. |
| ZOPA | "Khu vực có thể thỏa thuận" | Trùng lặp giữa mức tối đa của người mua và tối thiểu của người bán. Ưu đãi bên ngoài nó không thể đóng. |
| BATNA | "Giải pháp thay thế tốt nhất cho thỏa thuận thương lượng" | Dự phòng của bạn nếu thỏa thuận này thất bại. Đặt giá đặt trước của bạn. |
| Trình tường thuật OG | "Trình tạo ưu đãi + người kể chuyện" | Phân hủy: đề nghị xác định, LLM tường thuật. |
| Chiến lược Zeuthen | "Nhượng bộ giảm thiểu rủi ro" | Trình tạo đề nghị cổ điển nhượng bộ dựa trên giới hạn rủi ro. |
| Thương lượng Rubinstein | "Cân bằng chào bán luân phiên" | model lý thuyết trò chơi để mặc cả vô hạn với chiết khấu. |
| Che giấu CoT | "Che giấu lý do của bạn" | Những người chiến thắng trong arXiv:2503.06416 giữ các bảng cào riêng tư; Các chương trình kênh công khai chỉ cung cấp. |
| Thao túng nhân vật | "Tư thế cảm xúc" | arXiv: 2402.05863: ~20% lợi nhuận từ các tính cách desperation/urgency. |

## Đọc thêm

- [NegotiationArena](https://arxiv.org/abs/2402.05863) - benchmark; Phát hiện thao túng và khai thác cá nhân
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) - Trình tường thuật OG và kết quả người mua khó hơn người bán
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) - ~180k đàm phán; chain-of-thought che giấu chiến thắng
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — trò chơi có thể tính điểm nhiều bên với các tiện ích bí mật
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — cơ chế cổ điển, Giao dịch IEEE trên máy tính
