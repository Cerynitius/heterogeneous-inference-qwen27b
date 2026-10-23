#!/usr/bin/env python3
"""Collect the model's own output token ids for draft-vocab selection.

96 diverse prompts (zh/en chat, writing, math, code in several languages, translation, structured output, agent-style);
none of them is in the held-out acceptance prompts. Thinking on, official
sampling (generation_config), max_tokens = max_model_len - prompt sent explicitly; the client keeps the first
--tokens tokens of each answer and disconnects (vLLM aborts). Token ids via logprobs (return_tokens_as_token_ids).
Output: vocab/gen_<tag>.jsonl, one line per prompt {"i", "cat", "ids"}.
  vocab_freq.py --base http://127.0.0.1:8001 --tag w5 --conc 6 --tokens 2048
"""
import argparse, json, os, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)
os.environ["no_proxy"] = "*"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--tag", required=True)
ap.add_argument("--conc", type=int, default=6)
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--max-model-len", type=int, default=262144)
a = ap.parse_args()

P = {
 "zh_chat": ["周末想带父母去近郊玩一天，帮我规划一下行程，预算一千元以内。", "为什么冬天开车前要先热车？现在的车还需要吗？",
             "我最近总是失眠，有什么不吃药的改善办法？", "解释一下什么是通货膨胀，以及它对普通人存款的影响。",
             "猫和狗哪个更适合独居的上班族养？给出理由。", "怎么跟领导提加薪比较合适？给我一个沟通框架。",
             "介绍一下秦始皇统一六国的主要原因。", "北京和上海的生活成本对比，从房租、交通、饮食三方面说说。",
             "如何判断一个西瓜熟没熟？", "第一次去日本旅游需要注意哪些礼仪？", "给一个刚毕业的程序员一些理财建议。",
             "为什么天空是蓝色的，而日落时是红色的？"],
 "zh_write": ["写一篇三百字左右的散文，题目是《雨夜》。", "以客服身份回复一位抱怨快递延误三天的顾客，语气诚恳。",
              "帮我写一份产品发布会的开场白，产品是一款智能手表。", "写一个科幻微小说，主角是一台失去记忆的机器人。",
              "把下面这句话改写得更正式：这个方案我觉得不太行，得再改改。", "写一封辞职信，原因是想去读研究生。",
              "为一家社区咖啡馆写三条朋友圈文案。", "写一首现代诗，主题是城市里的流浪猫。",
              "帮我写年终总结的提纲，岗位是数据分析师。", "给小学生写一段介绍恐龙灭绝的科普短文。",
              "写一段相声开场的贯口，主题是程序员加班。", "以《一封给十年后自己的信》为题写一篇短文。"],
 "math": ["求解方程 x^3 - 6x^2 + 11x - 6 = 0，并说明方法。", "一个袋子里有3个红球5个蓝球，不放回地取两个，两个颜色相同的概率是多少？",
          "Prove that there are infinitely many primes.", "计算定积分 ∫_0^1 x·e^x dx。",
          "A train leaves at 60 km/h and another 2 hours later at 90 km/h from the same station. When does the second catch up?",
          "用数学归纳法证明 1+2+...+n = n(n+1)/2。", "What is the expected number of rolls of a fair die to see all six faces?",
          "矩阵 [[2,1],[1,2]] 的特征值和特征向量是什么？", "证明：任意三角形内角和为180度。",
          "Find all integer solutions of x^2 - y^2 = 45.", "一个数列满足 a1=1, a(n+1)=2a(n)+1，求通项公式。",
          "Explain the difference between permutation and combination with an example."],
 "code_py": ["用 Python 写一个线程安全的 LRU 缓存，并写单元测试。", "Write a Python script that parses an nginx access log and prints the top 10 IPs.",
             "解释 Python 的 GIL，以及 asyncio 和多进程分别适合什么场景，附示例代码。", "用 pandas 读取一个 CSV，按月份统计销售额并画折线图。",
             "Implement Dijkstra's algorithm in Python with a heap and explain its complexity.", "写一个 FastAPI 服务，支持用户注册和 JWT 登录。",
             "Refactor this code to be more Pythonic: for i in range(len(xs)): if xs[i] > 0: out.append(xs[i]*2)",
             "用 PyTorch 实现一个两层 MLP 训练 MNIST，包含训练循环。", "写一个装饰器，统计函数调用耗时并在超过阈值时打日志。",
             "How do I profile memory usage of a Python program? Show tools and code.", "用 Python 实现快速排序和归并排序并比较性能。",
             "写一个命令行工具，递归查找目录下重复的文件（按哈希）。"],
 "code_other": ["Write a C++17 thread pool with a task queue and futures.", "用 Rust 实现一个简单的 TCP echo server，使用 tokio。",
                "写一个 SQL 查询：找出每个部门工资最高的前三名员工。", "Write a bash script that backs up a directory to a tarball with a date stamp and keeps the last 7.",
                "用 TypeScript 写一个 React 组件，实现带搜索过滤的表格。", "Explain CUDA shared memory bank conflicts with an example kernel.",
                "用 Go 写一个并发爬虫，限制最大并发数为 10。", "Write a Dockerfile for a Node.js app with multi-stage build.",
                "写一个 Makefile，编译 src 目录下所有 .c 文件并链接成可执行文件。", "Implement a lock-free stack in C++ using std::atomic.",
                "用 JavaScript 实现防抖和节流函数并解释区别。", "Write a Triton kernel for row-wise softmax and explain the tiling."],
 "en_chat": ["What are the main causes of the French Revolution?", "Give me a 3-day itinerary for Kyoto in autumn.",
             "How does a transformer neural network work? Explain to a software engineer.", "What should I consider when buying a used car?",
             "Summarize the plot of Hamlet in five sentences.", "Why do cats purr?", "Compare renting vs buying a home in a high-cost city.",
             "Explain how vaccines train the immune system.", "Write a polite email declining a job offer.",
             "What are good habits for remote work productivity?", "Describe the water cycle for a 10-year-old.",
             "What is the difference between TCP and UDP?"],
 "translate": ["把下面这段翻译成英文：人工智能正在深刻改变各行各业的生产方式，但也带来了就业和隐私方面的新挑战。",
               "Translate into Chinese: The committee postponed its decision until more data from the field trials became available.",
               "把这句古文翻译成现代汉语：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
               "Translate to Japanese and Chinese: Thank you for your patience while we resolve this issue.",
               "把这段技术文档翻译成英文：该接口在高并发场景下会出现连接池耗尽的问题，建议调大最大连接数并增加超时重试。",
               "Translate into Chinese and keep the formatting: - Step 1: Install the package\n- Step 2: Run the migration\n- Step 3: Restart the service"],
 "struct": ["输出一个 JSON，描述三本书的书名、作者、出版年和评分。", "Create a markdown table comparing Python, Go and Rust on speed, safety and ecosystem.",
            "把以下信息整理成 YAML 配置：服务名 api，端口 8080，副本数 3，环境变量 DEBUG=false。", "Generate a JSON schema for a user profile with name, email, age and tags.",
            "用表格列出中国四大名著的作者、朝代和主要人物。", "Give me a CSV with 5 rows of fake sales data: date, region, product, units, revenue."],
 "agent": ["你是一个运维助手。服务器磁盘使用率 95%，请一步步排查并给出要执行的命令。", "Plan the steps to migrate a PostgreSQL database to a new server with minimal downtime.",
           "你需要调用天气工具查询北京明天的天气，先说明你会调用什么工具、传什么参数。", "Debug this error: ModuleNotFoundError: No module named 'torch' inside a conda env that has torch installed.",
           "设计一个电商系统的订单服务，说明数据库表结构和关键接口。", "Review this git workflow and suggest improvements: everyone commits directly to main."],
 "reason": ["有三个开关对应楼上三盏灯，你只能上楼一次，如何判断哪个开关控制哪盏灯？", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain.",
            "一个人说'我在说谎'，这句话是真是假？分析一下这个悖论。", "You have 8 balls, one heavier. Find it with two weighings on a balance.",
            "甲乙丙三人中只有一人说真话，甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？",
            "Estimate how many piano tuners there are in Chicago. Show your reasoning."],
}
items = [(c, p) for c, ps in P.items() for p in ps]
os.makedirs("vocab", exist_ok=True)
out = open(f"vocab/gen_{a.tag}.jsonl", "w")
lock = threading.Lock()


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def run(i):
    cat, p = items[i]
    msgs = [{"role": "user", "content": p}]
    n = json.load(post("/tokenize", {"model": a.model, "messages": msgs, "add_generation_prompt": True}))["count"]
    body = {"model": a.model, "messages": msgs, "stream": True, "logprobs": True, "top_logprobs": 0,
            "return_tokens_as_token_ids": True, "max_tokens": a.max_model_len - n}
    ids = []
    try:
        with post("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                for ch in d.get("choices") or []:
                    for c in ((ch.get("logprobs") or {}).get("content") or []):
                        ids.append(int(c["token"].split(":")[1]))
                if len(ids) >= a.tokens:
                    break
    except Exception as e:
        print(f"#{i} {type(e).__name__}: {e}", flush=True)
    with lock:
        out.write(json.dumps({"i": i, "cat": cat, "ids": ids[:a.tokens]}) + "\n"); out.flush()
    return len(ids[:a.tokens])


with ThreadPoolExecutor(a.conc) as ex:
    lens = list(ex.map(run, range(len(items))))
print(f"[{a.tag}] vocab_freq: {len(items)} prompts, {sum(lens)} tokens, short answers {sum(1 for x in lens if x < a.tokens)}", flush=True)
