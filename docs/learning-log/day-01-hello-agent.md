# Day 1 — What Is an AI Agent?

**Date:** 2026-07-13
**Study time:** 3 hours
# Chapter 1
## My definition

An AI agent is: An framework which have the ability to sensor the enviroment and will react the changes in this enviroment, After sensor the enviroment change,it will react the that change and start doing things.

### PEAS model (Smart traveling agent)
- Performance:try our best to make the most of the agent
- enviroment: weather forcaste and other api
- Acutor: functions to call api
- Sensors: analyze user's input 


## The agent loop
1. The first stage is perception , analyze user's input and send to next stage
2. Second stage is thought, thought can be divide to two part plan and tool selection
3. Once the Agent receive user's input, it will plan what to do next
4. Afterthat Agent will choose the right tools to handle the user request
5. Finally take action (call one of the tool in tools list )
6. The enviroment can feel the changes 
7. The changes is a new observation to the agent
8. Agent will start over from 1 to 5

## simple Agent

### System prompt
```pythpn
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""
```

### Agent loop
```python
import re

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL_ID = "YOUR_MODEL_ID"
TAVILY_API_KEY="YOUR_Tavily_KEY"
os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_API_KEY"

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
```

## 习题

> <strong>提示</strong>：以下的部分习题没有标准答案，重点在于培养学习者对智能体系统批判性的深入思考和动手实践能力。

1. 请分析以下四个 `case` 中的<strong>主体</strong>是否属于智能体，如果是，那么属于哪种类型的智能体（可以从多个分类维度进行分析），并说明理由：

   `case A`：<strong>一台符合冯·诺依曼结构的超级计算机</strong>，拥有高达每秒 2EFlop 的峰值算力

   `case B`：<strong>特斯拉自动驾驶系统</strong>在高速公路上行驶时，突然检测到前方有障碍物，需要在毫秒级做出刹车或变道决策

   `case C`：<strong>AlphaGo</strong>在与人类棋手对弈时，需要评估当前局面并规划未来数十步的最优策略

   `case D`：<strong>ChatGPT 扮演的智能客服</strong>在处理用户投诉时，需要查询订单信息、分析问题原因、提供解决方案并安抚用户情绪

2. 假设你需要为一个"智能健身教练"设计任务环境。这个智能体能够：
   - 通过可穿戴设备监测用户的心率、运动强度等生理数据
   - 根据用户的健身目标（减脂/增肌/提升耐力）动态调整训练计划
   - 在用户运动过程中提供实时语音指导和动作纠正
   - 评估训练效果并给出饮食建议

   请使用 PEAS 模型完整描述这个智能体的任务环境，并分析该环境具有哪些特性（如部分可观察、随机性、动态性等）。

3. 某电商公司正在考虑两种方案来处理售后退款申请：
   
   方案 A（`Workflow`）：设计一套固定流程，例如：

   A.1 对于一般商品且在 7 天之内，金额 `< 100RMB` 自动通过；`100-500RMB `由客服审核；`>500RMB` 需主管审批；而特殊商品（如定制品）一律拒绝退款
   
   A.2 对于超过 7 天的商品，无论金额，只能由客服审核或主管审批；
   
   方案 B（`Agent`）：搭建一个智能体系统，让它理解退款政策、分析用户历史行为、评估商品状况，并自主决策是否批准退款
   
   请分析：
   - 这两种方案各自的优缺点是什么？
   - 在什么情况下 `Workflow` 更合适？什么情况下 `Agent` 更有优势？如果你是该电商公司的负责人，你更倾向于采用哪种方案？
   - 是否存在一个方案 C，能够结合两种方案，达到扬长避短的效果？
   
4. 在 1.3 节的智能旅行助手基础上，请思考如何添加以下功能（可以只描述设计思路，也可以进一步尝试代码实现）：

   > <strong>提示</strong>：思考如何修改 `Thought-Action-Observation` 循环来实现这些功能。

   - 添加一个"记忆"功能，让智能体记住用户的偏好（如喜欢历史文化景点、预算范围等）
   - 当推荐的景点门票已售罄时，智能体能够自动推荐备选方案
   - 如果用户连续拒绝了 3 个推荐，智能体能够反思并调整推荐策略

5. 卡尼曼的"系统 1"（快速直觉）和"系统 2"（慢速推理）理论<sup>[2]</sup>为神经符号主义 AI 提供了很好的类比。请首先构思一个具体的智能体的落地应用场景，然后说明场景中的：

   > <strong>提示</strong>：医疗诊断助手、法律咨询机器人、金融风控系统等都是常见的应用场景

   - 哪些任务应该由"系统 1"处理？
   - 哪些任务应该由"系统 2"处理？
   - 这两个系统如何协同工作以达成最终目标？

6. 尽管大语言模型驱动的智能体系统展现出了强大的能力，但它们仍然存在诸多局限。请分析以下问题：
   - 为什么智能体或智能体系统有时会产生"幻觉"（生成看似合理但实际错误的信息）？
   - 在 1.3 节的案例中，我们设置了最大循环次数为 5 次。如果没有这个限制，智能体可能会陷入什么问题？
   - 如何评估一个智能体的"智能"程度？仅使用准确率指标是否足够？


## What I observed in FirstAgentTest.py

## One surprising observation

## One unresolved question

## Tomorrow's plan



# Chapter 3

## Tokens
Definition 