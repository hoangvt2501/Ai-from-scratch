# Theo dõi trạng thái đối thoại

> "Tôi muốn một nhà hàng giá rẻ ở phía bắc... thực sự làm cho nó vừa phải... và thêm tiếng Ý." Ba lượt, ba cập nhật trạng thái. DST giữ cho chính sách giá trị vị trí được đồng bộ hóa để đặt phòng hoạt động.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 5 · 17 (Chatbots), Giai đoạn 5 · 20 (Đầu ra có cấu trúc)
**Thời lượng:** ~75 phút

## Vấn đề

Trong hệ thống đối thoại theo hướng nhiệm vụ, mục tiêu của người dùng được mã hóa dưới dạng một tập hợp các cặp giá trị vị trí: `{cuisine: italian, area: north, price: moderate}`. Mỗi lượt người dùng có thể thêm, thay đổi hoặc xóa một vị trí. Hệ thống phải đọc toàn bộ cuộc trò chuyện và xuất trạng thái hiện tại một cách chính xác.

Đặt sai một vị trí và hệ thống đặt nhầm nhà hàng, lên lịch sai chuyến bay hoặc tính phí sai thẻ. DST là bản lề giữa những gì người dùng nói và những gì phụ trợ thực thi.

Tại sao nó vẫn quan trọng vào năm 2026 bất chấp LLMs:

- Các miền nhạy cảm với tuân thủ (ngân hàng, chăm sóc sức khỏe, đặt vé máy bay) yêu cầu các giá trị vị trí xác định, không phải tạo dạng tự do.
- agents sử dụng công cụ vẫn cần độ phân giải khe cắm trước khi gọi APIs.
- Điều chỉnh nhiều lượt khó hơn vẻ ngoài của nó: "thực sự không, hãy làm thứ Năm."

pipeline hiện đại: khái niệm DST cổ điển + bộ trích xuất LLM + guardrails đầu ra có cấu trúc.

## Khái niệm

![DST: dialog history → slot-value state](../assets/dst.svg)

**Cấu trúc nhiệm vụ.** schema xác định các miền (nhà hàng, khách sạn, taxi) và các vị trí của chúng (ẩm thực, khu vực, giá cả, con người). Mỗi khe có thể trống, được lấp đầy bằng một giá trị từ một tập hợp đóng (giá: {rẻ, vừa phải, đắt}) hoặc một giá trị dạng tự do (tên: "Ấm đun nước bằng đồng").

**Hai công thức DST.**

- **Phân loại.** Đối với mỗi cặp (vị trí, candidate_value), dự đoán yes/no. Hoạt động cho các vị trí từ vựng đóng. Tiêu chuẩn trước năm 2020.
- **Generation.** Với cuộc đối thoại, hãy tạo các giá trị vị trí dưới dạng văn bản tự do. Hoạt động cho các khe hở từ vựng. Mặc định hiện đại.

**Metric.** Joint Goal Accuracy (JGA) — phần số lượt mà *mỗi* vị trí là chính xác. Tất cả hoặc không có gì. Bảng xếp hạng MultiWOZ 2.4 đứng đầu khoảng 83% vào năm 2026.

**Kiến trúc.**

1. **Dựa trên quy tắc (biểu thức chính quy vị trí + từ khóa).** Đường cơ sở mạnh mẽ cho các miền hẹp. Có thể gỡ lỗi.
2. **TripPy / BERT-DST.** Tạo dựa trên bản sao với mã hóa BERT. Tiêu chuẩn LLM trước.
3. **LDST (LLaMA + LoRA).** LLM được điều chỉnh theo hướng dẫn với prompting khe miền. Đạt chất lượng cấp ChatGPT trên MultiWOZ 2.4.
4. **Không có bản thể học (2024–26).** Bỏ qua schema; Tạo tên và giá trị vị trí trực tiếp. Xử lý các miền mở.
5. **Prompt + đầu ra có cấu trúc (2024–26).** LLM với Pydantic schema + giải mã hạn chế. 5 dòng mã, sẵn sàng production.

### Các chế độ lỗi cổ điển

- **Đồng tham khảo qua các lượt.** "Hãy ở lại với lựa chọn đầu tiên." Cần giải quyết tùy chọn nào.
- **Ghi đè so với thêm vào.** Người dùng nói "thêm tiếng Ý". Bạn thay thế ẩm thực hay thêm vào?
- **Xác nhận ngầm.** "OK tuyệt" - điều đó có chấp nhận đặt phòng được đề nghị không?
- **Sửa chữa.** "Thực sự làm cho nó 7 giờ tối." Phải cập nhật thời gian mà không cần xóa các vị trí khác.
- **Đồng tham chiếu đến lời nói của hệ thống trước đó.** "Vâng, cái đó." "Cái đó" nào?

## Tự xây dựng

### Bước 1: trình trích xuất khe dựa trên quy tắc

Xem `code/main.py`. Từ điển Regex + từ đồng nghĩa bao gồm 70% các câu nói chính tắc trong các miền hẹp:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Giòn bên ngoài từ vựng kinh điển. Hoạt động để xác nhận vị trí xác định.

### Bước 2: vòng lặp cập nhật trạng thái

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Ba bất biến:

- Không bao giờ đặt lại khe cắm mà người dùng không chạm vào.
- Phủ định rõ ràng ("đừng bận tâm đến ẩm thực") phải rõ ràng.
- Sửa lỗi người dùng ("thực tế...") phải ghi đè, không được thêm vào.

### Bước 3: DST điều khiển LLM với đầu ra có cấu trúc

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic đảm bảo một đối tượng trạng thái hợp lệ. Không có regex, không có schema không khớp, không có máy đánh bạc ảo giác.

### Bước 4: Đánh giá JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Hiệu chỉnh: hệ thống nhận được TẤT CẢ các vị trí đúng ở phần nào của lượt? Đối với MultiWOZ 2.4, các hệ thống hàng đầu năm 2026: 80-83%. Hệ thống trong miền của bạn sẽ vượt quá mức đó về vốn từ vựng hẹp của bạn hoặc đường cơ sở LLM đánh bại bạn.

### Bước 5: xử lý hiệu chỉnh

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Khi hiệu chỉnh được phát hiện, hãy ghi đè lên vị trí được cập nhật gần đây nhất thay vì thêm vào. Khó có thể làm đúng nếu không có sự giúp đỡ của LLM. Mô hình hiện đại: luôn để LLM tái tạo toàn bộ trạng thái từ lịch sử thay vì cập nhật dần dần - điều này tự nhiên xử lý các chỉnh sửa.

## Cạm bẫy

- **Chi phí tái tạo toàn bộ lịch sử.** Để trạng thái tái tạo LLM mỗi lượt tiêu tốn tổng tokens O(n²). Giới hạn lịch sử hoặc tóm tắt các lượt cũ hơn.
- **Schema trôi dạt.** Thêm các vị trí mới sau khi phá vỡ dữ liệu training cũ. Phiên bản schema của bạn.
- **Phân biệt chữ hoa chữ thường.** "Ý" vs "Ý" vs "Ý" - chuẩn hóa ở mọi nơi.
- **Kế thừa ngầm.** Nếu trước đó người dùng đã chỉ định "cho 4 người", thì yêu cầu mới cho một thời gian khác sẽ không xóa mọi người. Luôn vượt qua toàn bộ lịch sử.
- **Dạng tự do so với tập hợp đóng.** Tên, thời gian và địa chỉ cần các vị trí dạng tự do; các món ăn và khu vực đóng cửa. Trộn cả hai trong schema.

## Ứng dụng

stack năm 2026:

| Tình huống | Cách tiếp cận |
|-----------|----------|
| Miền hẹp (một hoặc hai ý định) | Dựa trên quy tắc + biểu thức chính quy |
| Tên miền rộng, dữ liệu được gắn nhãn có sẵn | LDST (LLaMA + LoRA trên dữ liệu kiểu MultiWOZ) |
| Tên miền rộng, không có nhãn, sẵn sàng sản xuất | LLM + Người hướng dẫn + schema Pydantic |
| Nói / giọng nói | ASR + bộ chuẩn hóa + LLM-DST |
| Quy trình đặt phòng đa miền | LLM có hướng dẫn Schema với models Pydantic trên mỗi miền |
| Nhạy cảm với tuân thủ | Sơ cấp dựa trên quy tắc, dự phòng LLM với luồng xác nhận |

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Bài tập

1. **Dễ dàng.** Xây dựng trình theo dõi trạng thái dựa trên quy tắc trong `code/main.py` cho 3 vị trí (ẩm thực, khu vực, giá cả). Thử nghiệm trên 10 đoạn hội thoại thủ công. Đo lường JGA.
2. **Trung bình.** Cùng dataset với Huấn luyện viên + Pydantic + một LLM nhỏ. So sánh JGA. Kiểm tra những khúc cua khó nhất.
3. **Khó.** Triển khai cả hai và định tuyến: chính dựa trên quy tắc, LLM dự phòng khi dựa trên quy tắc tự tin phát ra các vị trí <2. Đo lường JGA kết hợp và chi phí inference mỗi lượt.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| DST | Theo dõi trạng thái hộp thoại | Duy trì vị trí giá trị qua các lượt đối thoại. |
| Khe cắm | Đơn vị ý định của người dùng | Được đặt tên parameter nhu cầu phụ trợ (ẩm thực, ngày). |
| Tên miền | Khu vực nhiệm vụ | Nhà hàng, khách sạn, taxi - bộ máy đánh bạc. |
| JGA | Mục tiêu chung Accuracy | Phần số lượt mà mọi vị trí đều chính xác. Tất cả hoặc không có gì. |
| Đa WOZ | Các benchmark | dataset WOZ đa miền; đánh giá DST tiêu chuẩn. |
| DST không có bản thể | Không schema | Tạo tên và giá trị vị trí trực tiếp, không có danh sách cố định. |
| Sửa chữa | "Thật ra..." | Turn that ghi đè lên một vị trí đã lấp đầy trước đó. |

## Đọc thêm

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) - benchmark kinh điển.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) - Điều chỉnh hướng dẫn LLaMA + LoRA cho DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) - con ngựa DST dựa trên bản sao.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) - TOD không giám sát dựa trên EM.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — kết quả DST chính tắc.
