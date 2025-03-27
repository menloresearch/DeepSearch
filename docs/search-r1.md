# Search-R1

- **WAIT WHAT? THIS ONLY USE 1 REWARD FUNCTION? 🤯** (outcome-based reward function - Exactmatch)
- Still required the model to generate xml structured output, but does not have a reward function to check the format.
- [ ] Develop deepsearch further from this project. The code is very detailed and well-written.
- <https://github.com/PeterGriffinJin/Search-R1>
- <https://arxiv.org/pdf/2503.09516>
- Trained a 3B qwen model with GRPO and multi hop tool call ability
- Reproduce the paper: <https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa>
- Apache-2.0 license

# Summary Key Points with NotebookLM

Dựa trên các nguồn, SEARCH-R1 giới thiệu một **khung học tăng cường (RL) mới cho phép các mô hình ngôn ngữ lớn (LLMs) tự động xen kẽ quá trình suy luận với tương tác với công cụ tìm kiếm theo thời gian thực**. Mục tiêu chính là giúp LLMs **thu thập kiến thức bên ngoài và thông tin cập nhật một cách hiệu quả** để nâng cao khả năng suy luận và tạo văn bản của chúng.

- **Hỗ trợ truy xuất và suy luận nhiều lượt**, trong đó các lệnh gọi tìm kiếm được kích hoạt rõ ràng bằng các mã thông báo `<search>` và `</search>`, còn nội dung được truy xuất được bao quanh bởi các mã thông báo `<information>` và `</information>`, và các bước suy luận của LLM được bao quanh bởi `<think>` và `</think>`, với câu trả lời cuối cùng được định dạng bằng `<answer>` và `</answer>`.
- **Áp dụng kỹ thuật che phủ mã thông báo được truy xuất (retrieved token masking)** để đảm bảo **tối ưu hóa RL ổn định**, bằng cách chỉ tính toán mục tiêu gradient chính sách trên các mã thông báo do LLM tạo ra và loại trừ nội dung được truy xuất khỏi quá trình tối ưu hóa.
- Sử dụng một **hàm thưởng đơn giản dựa trên kết quả cuối cùng (outcome-based reward function)**, đánh giá độ chính xác của câu trả lời cuối cùng, chẳng hạn như sử dụng so khớp chuỗi chính xác (Exact Match - EM) trong các tác vụ suy luận dựa trên факты. Hàm thưởng được định nghĩa là `rϕ(x, y) = EM(apred, agold)`. Thiết kế thưởng tối thiểu này được chứng minh là hiệu quả trong các tình huống tìm kiếm và suy luận.

**Về hàm thưởng và GRPO:**

- SEARCH-R1 sử dụng một **hệ thống thưởng dựa trên quy tắc chỉ bao gồm phần thưởng kết quả cuối cùng**. Điều này có nghĩa là mô hình chỉ được thưởng dựa trên việc câu trả lời cuối cùng của nó có đúng hay không so với đáp án thực tế. Các tác giả đã cố tình tránh sử dụng phần thưởng định dạng phức tạp hoặc huấn luyện các mô hình thưởng thần kinh (neural reward models) do lo ngại về việc "hack" phần thưởng và chi phí tính toán cũng như độ phức tạp gia tăng.
- SEARCH-R1 tương thích với nhiều thuật toán RL khác nhau, bao gồm cả **Proximal Policy Optimization (PPO)** và **Group Relative Policy Optimization (GRPO)**.
- **GRPO (Group Relative Policy Optimization)** là một phương pháp tối ưu hóa chính sách khác với PPO ở chỗ nó **sử dụng phần thưởng trung bình của nhiều đầu ra được lấy mẫu làm đường cơ sở (baseline)** thay vì dựa vào một hàm giá trị (value function) được học. Đối với mỗi câu hỏi đầu vào, GRPO lấy mẫu một nhóm phản hồi từ chính sách tham khảo (reference policy) và sau đó tối ưu hóa mô hình chính sách bằng cách tối đa hóa một hàm mục tiêu dựa trên phần thưởng tương đối trong nhóm.
- Nghiên cứu cho thấy rằng **GRPO thường hội tụ nhanh hơn PPO** vì PPO dựa vào một mô hình phê bình (critic model) cần một số bước khởi động trước khi quá trình huấn luyện hiệu quả bắt đầu. Tuy nhiên, **PPO thể hiện sự ổn định huấn luyện lớn hơn**, trong khi GRPO có thể dẫn đến sự sụp đổ phần thưởng trong một số trường hợp.
- Mặc dù có sự khác biệt về tốc độ hội tụ và độ ổn định, **phần thưởng huấn luyện cuối cùng của PPO và GRPO là tương đương nhau**.
- Kết quả đánh giá cho thấy **GRPO thường vượt trội hơn PPO** trong việc tối ưu hóa khả năng suy luận tăng cường bằng truy xuất. Ví dụ, trên cả Qwen2.5-3B và LLaMA3.2-3B, GRPO đạt được hiệu suất trung bình cao hơn.

## Training Templates

- As shown in Table 1, this template structures the model’s output into three parts in an iterative fashion: first, a reasoning process, then a search engine calling function, and finally, the answer.

## Reward modedling

- **rule-based** reward system that consists solely of final outcome rewards, which assess the **correctness of the model’s response**
- not use neural reward model cuz scared of reward hacking
- $r_\phi(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})$,
- a_pred is the **extracted final answer** from response y and a_gold is the ground truth answer
    - How to extract a_pred from response y with rule-based?

## Experiment setup

- For retrieval, we use the 2018 Wikipedia dump (Karpukhin et al., 2020) as the knowledge source and E5 (Wang et al., 2022) as the retriever.
- follow Lin et al. (2023) and set the number of retrieved passages to three across all retrieval-based methods.  
- Exact Match (EM) is used as the evaluation metric, following Yu et al. (2024) (Rankrag: Unifying context ranking with retrieval-augmented generation in llms)
    - just check the source code lol i'm lazy to read paper 💀
    - WAIT WHAT? why outcome EM came from a RAG paper?

```python
def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
```

## Datasets

- Training dataset:merge  training sets of NQ and HotpotQA
- Seven **benchmark**datasets,
    - General Question Answering: NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al., 2022).
    - Multi-Hop Question Answering: HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), Musique (Trivedi et al., 2022b), and Bamboogle (Press et al., 2022).

## Evaluation Baselines

- Inference without Retrieval: Direct inference and Chain-of-Thought (CoT) reasoning
- Inference with Retrieval: Retrieval-Augmented Generation (RAG)
- Finetune base models It only contains reasoning and answer steps and cannot call a search engine.

## Hotpot QA and NQ

### HotpotQA

- Size: ~113K crowd-sourced questions
- Type: Multi-hop question answering dataset
- Source: English Wikipedia
- Key Features:
    - Requires reading 2 Wikipedia articles to answer each question
    - Comes with gold paragraphs and supporting facts identified by crowdworkers
    - Diverse reasoning strategies including:
        - Questions with missing entities
        - Intersection questions ("What satisfies property A and B?")
        - Comparison questions (comparing entities by common attributes)

Two settings:

1. Few-document distractor: Models get 10 paragraphs containing the gold paragraphs
2. Open-domain fullwiki: Models only get the question and access to Wikipedia

Evaluation metrics:

- Answer accuracy: Exact Match (EM) and unigram F1
- Explainability: Supporting Fact EM/F1
- Joint metric for both tasks

### Natural Questions (NQ)

- Size: 300,000 questions
- Type: Open-domain question answering
- Source: Real Google search queries
- Key Features:
    - Natural questions from real users
    - Human-annotated answers from Wikipedia pages
    - Additional 16,000 examples with 5 different annotators per question for evaluation
    - Replicates end-to-end process of how people find answers

Example from HotpotQA (comparison type):

```
Question: "Which magazine was started first Arthur's Magazine or First for Women?"
Supporting Facts: 
- Arthur's Magazine was a literary periodical established in 1844
- First for Women is a woman's magazine launched in 1989
Answer: "Arthur's Magazine"
```
