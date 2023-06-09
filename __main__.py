from langchainme import OpenAI

openai_api_key = "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"

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

prompt = """
对下面内容做文本摘要.

文本内容:
上周，我有幸与Sam Altman和其他20名开发人员坐下来讨论OpenAI的API及其产品计划。Sam非常开放。讨论涉及了实际的开发人员问题，以及与OpenAI的使命和人工智能的社会影响相关的大问题。以下是关键的要点：
在整个讨论过程中出现的一个共同主题是，目前OpenAI的GPU非常有限，这推迟了他们的许多短期计划。客户最大的抱怨是关于API的可靠性和速度。Sam承认了他们的担忧，并解释说，大部分问题都是GPU短缺的结果。
更长的32k上下文还不能推广给更多的人。OpenAI尚未克服注意力机制的O(n^2)扩展问题，因此，尽管他们似乎有望在不久的将来（今年）提供10万到100万令牌的上下文窗口，但任何更大的窗口都需要研究上的突破。
微调API目前也受到GPU可用性的瓶颈。他们还没有使用适配器或LoRa等高效的微调方法，因此微调的运行和管理非常耗时需要计算。未来将有更好的微调支持。他们甚至可能举办一个社区贡献模型的市场。
专用容量供应受到GPU可用性的限制。OpenAI还提供专用容量，为客户提供该模型的私人副本。要访问此服务，客户必须愿意承诺预付10万美元。
"""

num_tokens = llm.get_num_tokens(prompt)
print(f"Our prompt has {num_tokens} tokens")

output = llm(prompt)
print(output)
