# Tại sao nên sử dụng Multi-Agent?

> Một agent đập vào tường. Động thái thông minh không phải là một agent lớn hơn - nó agents hơn.

**Loại:** Học
**Ngôn ngữ:** TypeScript
**Kiến thức tiên quyết:** Giai đoạn 14 (Kỹ thuật Agent)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xác định trần một agent (tràn ngữ cảnh, chuyên môn hỗn hợp, tắc nghẽn tuần tự) và giải thích khi chia thành nhiều agents là bước đi đúng đắn
- So sánh các mẫu orchestration (pipeline, phân xuất song song, giám sát, phân cấp) và chọn mẫu phù hợp cho một cấu trúc tác vụ nhất định
- Thiết kế hệ thống đa agent với ranh giới vai trò rõ ràng, trạng thái chia sẻ và hợp đồng giao tiếp
- Phân tích sự đánh đổi giữa độ phức tạp nhiều agent (độ trễ, chi phí, độ khó gỡ lỗi) so với đơn giản agent đơn

## Vấn đề

Bạn đã xây dựng một agent duy nhất trong Giai đoạn 14. Nó hoạt động. Nó có thể đọc tệp, chạy lệnh, gọi APIs và suy luận về kết quả. Sau đó, bạn chỉ nó vào một cơ sở mã thực sự: 200 tệp, ba ngôn ngữ, các bài kiểm tra phụ thuộc vào cơ sở hạ tầng và yêu cầu nghiên cứu APIs bên ngoài trước khi viết mã.

agent nghẹt thở. Không phải vì LLM ngu ngốc, mà vì nhiệm vụ vượt quá những gì một vòng lặp agent có thể xử lý. context window chứa đầy nội dung tệp. agent quên những gì nó đã đọc 40 lần gọi công cụ trước. Nó cố gắng trở thành một nhà nghiên cứu, một lập trình viên và một người đánh giá cùng một lúc, và làm cả ba đều kém.

Đây là trần một agent. Bạn nhấn nó mỗi khi một nhiệm vụ yêu cầu:

- **Nhiều ngữ cảnh hơn là phù hợp với một cửa sổ** - đọc 50 tệp vượt quá 200k tokens
- **Chuyên môn khác nhau ở các giai đoạn khác nhau** - nghiên cứu đòi hỏi prompting khác với việc tạo mã
- **Công việc có thể diễn ra song song** - tại sao phải đọc ba tệp tuần tự khi bạn có thể đọc chúng cùng một lúc?

## Khái niệm

### Trần một Agent

Một agent duy nhất là một vòng lặp, một context window, một system prompt. Hãy hình dung nó:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Ba điều gặp lỗi:

1. **Độ bão hòa ngữ cảnh** - kết quả công cụ chồng chất. Đến lượt 30, agent đã tiêu thụ 150k tokens nội dung tệp, đầu ra lệnh và lý luận prior. Các chi tiết quan trọng từ lượt 5 bị mất.

2. **Nhầm lẫn vai trò** - một system prompt nói rằng "bạn là một nhà nghiên cứu, lập trình viên, người đánh giá và người kiểm tra" tạo ra một agent nửa nghiên cứu, nửa mã và không bao giờ hoàn thành việc xem xét.

3. **Tắc nghẽn tuần tự** - agent đọc tệp A, sau đó tệp B, sau đó tệp C. Ba cuộc gọi LLM nối tiếp. Ba lần thực thi công cụ nối tiếp. Không song song.

### Giải pháp đa Agent

Chia nhỏ công việc. Cung cấp cho mỗi agent một công việc, một context window và một system prompt được điều chỉnh cho công việc đó:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Mỗi agent có:
- Một system prompt tập trung ("Bạn là một người đánh giá mã. Công việc duy nhất của bạn là tìm lỗi.")
- context window riêng của nó (không bị ô nhiễm bởi các tác phẩm của agents khác)
- Hợp đồng input/output rõ ràng (nhận ghi chú nghiên cứu, mã đầu ra)

### Hệ thống thực tế làm điều này

**Claude Code subagents** - khi Claude Code tạo ra một subagent với `Task`, nó sẽ tạo ra một agent con với một nhiệm vụ có phạm vi. Cha mẹ giữ ngữ cảnh của nó sạch sẽ. Đứa trẻ làm bài tập trung và trả về một bản tóm tắt.

**Devin** - chạy agent lập kế hoạch, agent lập trình và agent trình duyệt. Người lập kế hoạch chia công việc thành các bước. Lập trình viên viết mã. Trình duyệt nghiên cứu tài liệu. Mỗi loại có bối cảnh riêng biệt.

**Nhóm mã hóa đa agent (SWE-bench)** - các hệ thống hoạt động hàng đầu trên SWE-bench sử dụng một nhà nghiên cứu đọc cơ sở mã, một người lập kế hoạch thiết kế bản sửa lỗi và một lập trình viên thực hiện nó. Hệ thống một agent điểm thấp hơn.

**ChatGPT Nghiên cứu sâu** - tạo ra nhiều agents tìm kiếm song song, mỗi  khám phá một góc độ khác nhau, sau đó tổng hợp kết quả.

### Quang phổ

Multi-agent không phải là nhị phân. Nó là một quang phổ:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**agent đơn **- một vòng lặp, một prompt. Tốt cho các tác vụ đơn giản.

**Subagents** - cha mẹ sinh ra con cho các nhiệm vụ phụ tập trung. Cha mẹ duy trì kế hoạch. Trẻ em báo cáo lại. Đây là những gì Claude Code làm.

**Pipeline** - agents chạy theo trình tự. Agent đầu ra của A trở thành đầu vào của Agent B. Tốt cho quy trình làm việc theo giai đoạn: nghiên cứu -> code -> review -> test.

**Nhóm** - agents chạy song song với bus tin nhắn được chia sẻ. Mỗi người có một vai trò. Một người phối hợp tọa độ. Tốt khi cần đồng thời các skills khác nhau.

**Swarm** - nhiều agents giống hệt hoặc gần giống hệt nhau với trạng thái chung. Không có bộ điều phối cố định. Agents nhận công việc từ hàng đợi. Tốt cho các tác vụ song song thông lượng cao.

### Bốn mô hình đa Agent

#### Mẫu 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Mỗi agent chuyển đổi dữ liệu và chuyển tiếp dữ liệu. Đơn giản để lý luận. Thất bại trong một giai đoạn chặn rest.

#### Mẫu 2: Quạt ra / Quạt vào

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Phân chia công việc trên các agents song song, sau đó merge kết quả. Tốt cho các nhiệm vụ phân tách thành các nhiệm vụ con độc lập.

#### Mẫu 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Một người điều phối thông minh quyết định phải làm gì, ủy quyền cho workers và tổng hợp kết quả. Bản thân trình điều phối là một agent với các công cụ để sinh sản workers.

#### Mẫu 4: Swarm ngang hàng

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Không có người điều phối trung tâm. Agents giao tiếp ngang hàng. Các quyết định xuất hiện từ sự tương tác. Khó gỡ lỗi hơn, nhưng có thể mở rộng thành nhiều agents.

### Khi nào KHÔNG sử dụng Multi-Agent

Multi-agent làm tăng thêm độ phức tạp. Mọi tin nhắn giữa agents đều là một điểm thất bại tiềm ẩn. Gỡ lỗi đi từ "đọc một cuộc trò chuyện" thành "trace tin nhắn trong năm agents".

**Ở agent độc thân khi:**
- Tác vụ phù hợp với một context window (dưới ~100k tokens dữ liệu làm việc)
- Bạn không cần prompts hệ thống khác nhau cho các giai đoạn khác nhau
- Thực hiện tuần tự đủ nhanh
- Nhiệm vụ đủ đơn giản để việc tách nó làm tăng thêm chi phí hơn giá trị

**Chi phí phức tạp:**
- Mỗi ranh giới agent là một bước nén mất dữ liệu: ngữ cảnh đầy đủ của agent A được tóm tắt thành một thông báo cho agent B
- Logic phối hợp (ai làm gì, khi nào, theo thứ tự nào) là nguồn lỗi của chính nó
- Độ trễ tăng: N agents có nghĩa là N cuộc gọi LLM nối tiếp tối thiểu, nhiều hơn nếu họ cần nói chuyện qua lại
- Chi phí nhân lên: mỗi agent đốt cháy tokens độc lập

Quy tắc chung: nếu một tác vụ mất ít hơn 20 lần gọi công cụ và phù hợp với 100k tokens, hãy giữ nó ở agent đơn.

```figure
swarm-messages
```

## Tự xây dựng

### Bước 1: Agent đơn quá tải

Đây là một agent duy nhất đang cố gắng làm mọi thứ. Nó có một system prompt lớn và một context window tổ chức nghiên cứu, mã và đánh giá:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Các vấn đề với cách tiếp cận này:
- context window phát triển theo từng giai đoạn. Theo bước xem xét, nó chứa các ghi chú nghiên cứu VÀ mã VÀ lý luận prior.
- Các system prompt là chung chung. Nó không thể được điều chỉnh cho từng giai đoạn.
- Không có gì chạy song song.

### Bước 2: Chuyên gia Agents

Bây giờ hãy tách nó ra. Mỗi agent nhận được một công việc:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Mỗi chuyên gia có một prompt tập trung. Mỗi người nhận được một context window sạch chỉ với đầu vào mà nó cần.

### Bước 3: Phối hợp thông qua tin nhắn

Kết nối các chuyên gia với nhau bằng cách truyền thông điệp rõ ràng:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Mỗi agent chỉ nhận được các tin nhắn được gửi đến nó. Không có ô nhiễm bối cảnh. 50k tokens đọc tài liệu của nhà nghiên cứu không bao giờ đi vào ngữ cảnh của người phản biện.

### Bước 4: So sánh

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Phiên bản đa agent sử dụng tổng số tokens hơn (ba agents, ba cuộc gọi LLM riêng biệt) nhưng ngữ cảnh của mỗi agent vẫn sạch sẽ. Chất lượng của từng công đoạn được cải thiện vì system prompt được chuyên biệt.

## Ứng dụng

Bài học này tạo ra một prompt có thể tái sử dụng để quyết định khi nào nên sử dụng nhiều agent. Xem `outputs/prompt-multi-agent-decision.md`.

## Bài tập

1. Thêm một chuyên gia thứ tư: một agent "tester" nhận mã từ lập trình viên và xem xét phản hồi từ người đánh giá, sau đó viết các bài kiểm tra
2. Sửa đổi pipeline để người đánh giá có thể gửi phản hồi trở lại lập trình viên để sửa đổi vòng lặp (tối đa 2 vòng)
3. Chuyển đổi pipeline tuần tự thành fan-out: chạy song song nhà nghiên cứu và "trình phân tích yêu cầu" agent, sau đó merge đầu ra của chúng trước khi chuyển đến mã hóa

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| Swarm | "Một tâm trí tổ ong của AI agents" | Một tập hợp các agents ngang hàng với trạng thái chung và không có nhà lãnh đạo cố định. Hành vi xuất hiện từ các tương tác cục bộ. |
| Điều phối viên | "Ông chủ agent" | Một agent có các công cụ bao gồm sinh sản và quản lý các agents khác. Nó lập kế hoạch và ủy quyền nhưng có thể không thực hiện công việc thực tế. |
| Điều phối viên | "Cảnh sát giao thông" | Một thành phần không agent (thường chỉ là mã, không phải LLM) định tuyến tin nhắn giữa các agents dựa trên các quy tắc. |
| Đồng thuận | "Các agents đồng ý" | Một giao thức trong đó nhiều agents phải đạt được thỏa thuận trước khi tiến hành. Được sử dụng khi các đầu ra xung đột cần giải quyết. |
| Hành vi nổi lên | "Các agents đã tự tìm ra nó" | Các mẫu cấp hệ thống phát sinh từ các tương tác agent nhưng không được lập trình rõ ràng. Có thể hữu ích hoặc có hại. |
| Quạt ra / quạt vào | "Giảm bản đồ cho agents" | Chia tác vụ trên các agents song song (fan-out), sau đó kết hợp kết quả của chúng (fan-in). |
| Truyền tin nhắn | "Agents nói chuyện với nhau" | Cơ chế giao tiếp giữa agents: dữ liệu có cấu trúc được gửi từ agent này sang  khác, thay thế windows ngữ cảnh được chia sẻ. |

## Đọc thêm

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - khảo sát các mô hình đa agent
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - framework hội thoại đa agent của Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - cách Claude Code ủy quyền với Task
- [CrewAI documentation](https://docs.crewai.com/) - đa agent framework dựa trên vai trò
