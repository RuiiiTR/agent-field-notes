# Day 5 — Plan and solve

**Date:** 2026-08-02
**Study time:** 2 hours


## Plan and Solve Definition
1. Planning Phase: Agent will make a plan, and list the things need to be done
2. Solving Phase: Agent will follow the plan and not changing the plan

## Why Plan and Solve
1. Coding a complex app
2. solving a hard math problem

## Plan and Solve comparison with othe method
Compare to ReAct, max step is n + 1(n determined by number of LLM api call)

## Plan and Solve process (my understanding)
The planner prompt goes out on its own first, and you wait for it to come back. Once you have the plan, your for loop starts running: for each step, you build a fresh prompt out of the executor template + question + plan + history + current step, send it on its own, wait for it to come back, append the result into history, then build the next one. That's n+1 independent requests in total, and no two of them are ever in flight at the same time.
The server has no idea it's the second or third call, so it does identical work every time.

Your template wraps the question in text. The SDK(The SDK handles: the URL, auth headers, JSON serialization, response parsing into objects, retries, timeouts, streaming, and typed errors.) wraps that text in JSON. The JSON is an envelope that the server opens and throws away; the text is the actual letter.
## Plan and Solve strenght and weakness
### Strenght
### Weakness
## Plan and Solve code representation
Planner prompt is only for the planner class, The planner class is a wrapper for the planner
prompt, IF you add some line of code to this Class, the result is not likely to change a lot.
```text
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
```
Class Planner is basiclly a prompt, the most important part is a prompt

```python
# 假定 llm_client.py 中的 HelloAgentsLLM 类已经定义好
# from llm_client import HelloAgentsLLM

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        
        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---")
        # 使用流式输出来获取完整的计划
        response_text = self.llm_client.think(messages=messages) or ""
        
        print(f"✅ 计划已生成:\n{response_text}")
        
        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

```
Solve class is wrapper for the execuction prompt, this prompt is only for the solve phase

```text
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""
```
This class is a wrapper for the EXECUTOR_PROMPT_TEMPLATE and the history variable will increment 
each round of plan list.



