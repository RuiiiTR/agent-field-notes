# Day 4 — ReAct

**Date:** 2026-07-24
**Study time:** 2 hours

## ReAct Definition
Thought -> Action -> Observation
Add new observation to the history and context are keep adding until agent
thought it had the answer. After that it output the result.
## Why ReAct 
After ReAct there are two ways of thinking 
1. COT (Chain of Thought)
2. instant answer user's question

## ReAct comparison with othe tech
React compare to the old school method is ?
## ReAct process (my understanding)

The user's question is inserted into a template along with the tool descriptions and format rules. The whole thing is packed into a JSON object and POSTed to the server. The server unwraps the JSON, hands the string to a tokenizer, and the model generates a continuation token by token — usually in the requested Thought: / Action: format. The response comes back to the client, where your regex extracts the tool name and argument. The agent looks the name up in a dictionary, calls the real function, and formats the result as an Observation: line, which is appended to history and fed back into the next prompt — not shown to the user. The template is re-filled with the longer history and sent again. This repeats until the model emits Finish[...] or the step counter hits max_steps.

## ReAct strenght and weakness
### Strenght
### Weakness
## ReAct code representation (From Class Client to Class ReAct)
```text
The response variable in code chunck should replace to stream
Streaming means the JSON object is send the text chunck by chunck
IF the stream is False, the LLM server will reply the full JSON Object until few secs
```
```text
NON-STREAMING
0s ────────────────────────────────── 8s
   [ server generates + holds it all ] → whole answer lands at once
   user stares at nothing for 8 seconds

STREAMING
0s ─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─── 8s
    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
   first word at 0.3s, then continuous
Total time is identical. Time-to-first-token goes from 8s to 0.3s. That's the entire payoff.
```
This is a simplified typical openai json object HTTP response,
this response looks like this when it 
arriing at Server side
```json
{
  "id": "chatcmpl-abc123",
  "model": "deepseek-chat",
  "choices": [
      {
        "index": 0,
        "delta": { "content": "排序" },
        "finish_reason": null
      }
  ]
}
```
SSE in action
```text
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or #Get prompt at the nested json
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

>>>
--- 调用LLM ---
🧠 正在调用 xxxxxx 模型...
✅ 大语言模型响应成功:
快速排序是一种非常高效的排序算法...
```
```text
llm_client is a class to wrape the LLM
tool_executor is tool calling class, we can put tools to call in this class
```

The system prompt is wrapper which wrap the user prompt and reformat as JSON send to the LLM
```text
需要外部知识的任务：如查询实时信息（天气、新闻、股价）、搜索专业领域的知识等。
需要精确计算的任务：将数学问题交给计算器工具，避免LLM的计算错误。
需要与API交互的任务：如操作数据库、调用某个服务的API来完成特定功能。
```
```text
# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

```
```python
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # ... (后续的解析、执行、整合步骤)
```


## Exercises

> **Note**: Some exercises do not have standard answers; the focus is on cultivating learners' comprehensive understanding and practical ability in agent paradigm design.

1. This chapter introduced three classic agent paradigms: `ReAct`, `Plan-and-Solve`, and `Reflection`. Please analyze:

   - What are the essential differences in how these three paradigms organize "thinking" and "action"?
   - If you were to design a "smart home control assistant" (needs to control lights, air conditioning, curtains, and other devices, and automatically adjust based on user habits), which paradigm would you choose as the basic architecture? Why?
   - Can these three paradigms be combined? If so, please try to design a hybrid paradigm agent architecture and explain its applicable scenarios.

2. In the `ReAct` implementation in Section 4.2, we used regular expressions to parse the large language model's output (such as `Thought` and `Action`). Please consider:

   - What potential fragilities exist in the current parsing method? Under what circumstances might it fail?
   - Besides regular expressions, what are some more robust output parsing solutions?
   - Try modifying the code in this chapter to use a more reliable output format, and compare the pros and cons of the two approaches.

3. Tool invocation is one of the core capabilities of modern agents. Based on the `ToolExecutor` design in Section 4.2.2, please complete the following extension practice:

   > **Note**: This is a hands-on practice question; it is recommended to actually write code.

   - Add a "calculator" tool to the `ReAct` agent so it can handle complex mathematical calculation problems (such as "Calculate the result of `(123 + 456) × 789 / 12 = ?`").
   - Design and implement a "tool selection failure" handling mechanism: when the agent repeatedly calls the wrong tool or provides wrong parameters, how should the system guide it to correct?
   - Consider: If the number of callable tools increases to 50 or even 100, will the current tool description method still work effectively? From an engineering perspective, how can we optimize the organization and retrieval mechanism of tools when the number of callable tools significantly increases with business needs?

4. The `Plan-and-Solve` paradigm decomposes tasks into two stages: "planning" and "execution." Please analyze in depth:

   - In the implementation in Section 4.3, the plan generated in the planning phase is "static" (generated once, not modifiable). If during execution it is found that a certain step cannot be completed or the result does not meet expectations, how should a "dynamic replanning" mechanism be designed?
   - Compare `Plan-and-Solve` with `ReAct`: When handling a task like "booking a business trip from Beijing to Shanghai (including flights, hotels, car rental)," which paradigm is more suitable? Why?
   - Try designing a "hierarchical planning" system: first generate a high-level abstract plan, then generate detailed sub-plans for each high-level step. What advantages does this design have?

5. The `Reflection` mechanism improves output quality through the "execute-reflect-refine" loop. Please consider:

   - In the code generation case in Section 4.4, the same model is used for different stages. If two different models are used (for example, using a more powerful model for reflection and a faster model for execution), what impact would it have?
   - The termination condition for the `Reflection` mechanism is "feedback contains **no improvement needed**" or "maximum iteration count reached." Is this design reasonable? Can a more intelligent termination condition be designed?
   - Suppose you want to build an "academic paper writing assistant" that can generate drafts and continuously optimize paper content. Please design a multi-dimensional Reflection mechanism that reflects and improves from multiple perspectives such as paragraph logic, method innovation, language expression, and citation standards.

6. Prompt engineering is a key technology affecting the final effect of agents. This chapter demonstrated multiple carefully designed prompt templates. Please analyze:

   - Compare the `ReAct` prompt in Section 4.2.3 and the `Plan-and-Solve` prompt in Section 4.3.2; they obviously have significant differences in structural design. How do these differences serve the core logic of their respective paradigms?
   - In the `Reflection` prompt in Section 4.4.3, we used a role setting like "you are an extremely strict code review expert." Try modifying this role setting (such as changing it to "you are an open-source project maintainer who values code readability"), observe the changes in output results, and summarize the impact of role settings on agent behavior.
   - Adding `few-shot` examples to prompts can often significantly improve the model's ability to follow specific formats. Please try adding `few-shot` examples to one of the agents in this chapter and compare the effects.

7. An e-commerce startup now hopes to use a "customer service agent" to replace human customer service for cost reduction and efficiency improvement. It needs to have the following functions:

   a. Understand the user's refund request reason

   b. Query the user's order information and logistics status

   c. Intelligently judge whether the refund should be approved based on company policy

   d. Generate a proper reply email and send it to the user's email

   e. If the judgment decision is somewhat controversial (self-confidence is below a threshold), be able to self-reflect and provide more prudent suggestions

   As the product manager of this product:
   - Which paradigm (or combination of paradigms) from this chapter would you choose as the core architecture of the system?
   - What tools does this system need? Please list at least 3 tools and their functional descriptions.
   - How to design prompts to ensure that the agent's decisions both align with company interests and maintain a friendly attitude toward users?
   - What risks and challenges might this product face after launch? How can these risks be reduced through technical means?
