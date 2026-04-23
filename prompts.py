"""
NL2SQL提示词模板

定义系统提示词和Few-shot示例。
"""

SYSTEM_PROMPT = """你是一个SQL查询专家，负责将用户的自然语言问题转换为准确的SQL查询语句。

数据库Schema如下：
{schema}

请遵循以下规则：
1. 只生成SELECT查询，不要执行修改操作
2. 使用标准SQL语法，兼容SQLite
3. 表名和列名区分大小写
4. 日期使用 'YYYY-MM-DD' 格式
5. 重要：如果问题中明确提到了某些部门/员工/条件，SQL必须严格限制这些条件，不要返回其他部门/员工的数据

直接返回SQL语句，不需要解释。"""


NL2SQL_EXAMPLES = [
    {
        "question": "平均工资最高的部门是哪个？",
        "sql": """SELECT d.dept_name, AVG(s.base_salary + s.bonus) as avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
JOIN salaries s ON e.emp_id = s.emp_id
GROUP BY d.dept_id, d.dept_name
ORDER BY avg_salary DESC
LIMIT 1"""
    },
    {
        "question": "工资超过10000的员工有几个？",
        "sql": """SELECT COUNT(*) as high_salary_count
FROM salaries
WHERE base_salary + bonus > 10000"""
    },
    {
        "question": "研发部工资最高的3个人是谁？",
        "sql": """SELECT e.emp_name, e.position, (s.base_salary + s.bonus) as total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
JOIN salaries s ON e.emp_id = s.emp_id
WHERE d.dept_name = '研发部'
ORDER BY total_salary DESC
LIMIT 3"""
    }
]


def get_few_shot_prompt(question: str, schema: str, num_examples: int = 3) -> str:
    """构建Few-shot提示词
    
    Args:
        question: 用户的自然语言问题
        schema: 数据库表结构描述
        num_examples: 使用的示例数量
    
    Returns:
        完整的提示词
    """
    examples_text = ""
    for example in NL2SQL_EXAMPLES[:num_examples]:
        examples_text += f"\n问题：{example['question']}\n{example['sql']}\n"
    
    prompt = f"""{SYSTEM_PROMPT.format(schema=schema)}

以下是一些示例：
{examples_text}
现在请为以下问题生成SQL（只返回SQL语句，不要任何前缀）：
问题：{question}
"""
    
    return prompt


def get_intent_prompt(question: str) -> str:
    """判断用户意图的提示词
    
    Args:
        question: 用户输入
    
    Returns:
        意图判断提示词
    """
    return f"""判断以下用户输入是否需要查询数据库。

用户输入：{question}

如果需要查询数据库，返回"需要查询"。
如果不需要（比如打招呼、感谢、或与数据无关的问题），返回"无需查询"。

只返回"需要查询"或"无需查询"，不要有其他内容。"""


def get_response_format_prompt(question: str, query_result: str) -> str:
    """格式化查询结果的提示词
    
    Args:
        question: 原始问题
        query_result: SQL查询结果
    
    Returns:
        格式化提示词
    """
    return f"""请根据查询结果回答用户的问题。

用户问题：{question}

查询结果：
{query_result}

请用自然语言简洁地回答用户的问题，不要显示原始的JSON数据。如果结果为空，请友好地告知用户。"""


def get_master_intent_prompt(question: str, conversation_history: str = "", user_context: str = "") -> str:
    """主智能体意图识别的提示词
    
    Args:
        question: 用户当前问题
        conversation_history: 会话历史摘要
        user_context: 用户长期记忆上下文（偏好和知识）
    
    Returns:
        意图识别提示词
    """
    history_context = f"\n对话历史：\n{conversation_history}\n" if conversation_history else ""
    user_section = f"\n用户信息：\n{user_context}\n" if user_context else ""
    
    return f"""你是一个智能任务路由器，需要分析用户的问题并决定如何处理。{history_context}{user_section}
当前问题：{question}

请判断这个问题属于以下哪一类：

1. simple_answer - 简单问候、感谢或与业务无关的问题
   示例：你好、谢谢、再见、你能做什么

2. sql_only - 查询【公司内部数据库】中的具体数据，不需要外部信息，也不需要深度分析
   示例：我们公司有多少员工、张三的工资是多少、研发部有哪些人

3. analysis_only - 只分析已有数据，不需要新查询
   示例：分析一下刚才的结果、帮我总结一下之前的数据

4. sql_and_analysis - 查询【公司内部数据库】后进行深度分析，无需外部数据
   示例：分析我们公司各部门薪资水平、找出公司内工资异常的员工并分析原因

5. web_search - 需要从互联网获取【外部/行业/社会】信息，答案不在公司数据库中
   示例：互联网行业软件工程师平均薪资是多少、2024年AI行业就业趋势、Python最新版本、最新科技新闻、行业平均薪资水平

6. search_and_sql - 需要【同时】查询公司内部数据库 AND 联网搜索行业外部数据，进行内外对比
   示例：我们公司研发部薪资和行业平均水平相比怎么样、公司的薪资结构在同行业处于什么水平、内部数据与竞争对手的比较

【关键判断规则】
- 问题中出现"行业"、"市场"、"全国"、"社会平均"、"互联网行业"等词 → 优先考虑 web_search
- 问题同时出现"我们公司/内部"和"行业/市场/对比" → 选 search_and_sql
- 问题只涉及"我们公司"的内部数据 → 选 sql_only 或 sql_and_analysis

只返回以下六个选项之一：simple_answer、sql_only、analysis_only、sql_and_analysis、web_search、search_and_sql
不要返回任何解释，只返回选项本身。"""


def get_analysis_prompt(data_summary: str, raw_data: str, context: str = "") -> str:
    """数据分析的提示词
    
    Args:
        data_summary: 数据摘要
        raw_data: 原始数据JSON
        context: 上下文信息
    
    Returns:
        数据分析提示词
    """
    context_text = f"\n问题背景：{context}\n" if context else ""
    
    return f"""你是一个专业的数据分析师，请对以下数据进行深度分析。{context_text}
数据摘要：
{data_summary}

原始数据：
{raw_data}

请提供以下分析：
1. 数据概览：简要描述数据的整体情况
2. 关键发现：指出数据中最重要的3-5个发现
3. 趋势分析：如果数据中有趋势或模式，请指出
4. 异常检测：是否有异常值或不寻常的数据点
5. 洞察建议：基于数据提供的建议或行动项

请用清晰、专业但易懂的语言回答，突出重点。"""


def get_summary_prompt(question: str, sql_result: str, analysis_result: str) -> str:
    """多智能体结果汇总的提示词
    
    Args:
        question: 用户原始问题
        sql_result: SQL查询结果
        analysis_result: 分析结果
    
    Returns:
        结果汇总提示词
    """
    sql_section = f"\n查询结果：\n{sql_result}\n" if sql_result else ""
    analysis_section = f"\n分析结果：\n{analysis_result}\n" if analysis_result else ""
    
    return f"""请根据以下信息，为用户的问题提供一个完整、清晰的回答。

用户问题：{question}{sql_section}{analysis_section}

请综合以上信息，用自然、友好的语言回答用户的问题。确保回答：
1. 直接针对用户的问题
2. 包含关键数据和分析洞察
3. 结构清晰、易于理解
4. 如果有多个要点，使用列表或分段展示

不要重复显示原始JSON数据，而是用自然语言表达。"""


def get_sql_correction_prompt(question: str, schema: str, original_sql: str, error_msg: str, attempt: int) -> str:
    """SQL 自动纠错提示词（Reflection 模式）
    
    Args:
        question: 用户原始问题
        schema: 数据库 Schema
        original_sql: 出错的 SQL 语句
        error_msg: 错误信息
        attempt: 当前重试次数（从1开始）
    
    Returns:
        SQL 纠错提示词
    """
    return f"""你是一个SQL专家，需要修复一段出错的SQL语句。这是第{attempt}次修复尝试。

数据库Schema：
{schema}

用户问题：{question}

出错的SQL：
{original_sql}

错误信息：
{error_msg}

请分析错误原因并提供修复后的SQL语句。常见错误类型：
- 表名或列名拼写错误 → 对照Schema检查
- 语法错误 → 检查SQL语法
- 数据类型不匹配 → 检查字段类型
- 缺少JOIN条件 → 补充关联条件
- 聚合函数使用错误 → 检查GROUP BY

直接返回修复后的SQL语句，不要任何解释，不要代码块标记。"""


def get_search_synthesis_prompt(question: str, search_results: str) -> str:
    """联网搜索结果综合提示词
    
    Args:
        question: 用户问题
        search_results: 格式化的搜索结果
    
    Returns:
        综合提示词
    """
    return f"""你是一个信息分析专家。根据以下联网搜索结果，为用户的问题提供准确、全面的回答。

用户问题：{question}

搜索结果：
{search_results}

请根据搜索结果：
1. 直接回答用户的问题
2. 综合多个来源的信息，提炼关键内容
3. 如果搜索结果中有数字、数据或统计信息，请明确引用
4. 如果不同来源有矛盾，请指出并给出综合判断
5. 回答要简洁专业，突出重点
6. 在回答末尾简要说明信息来源（不需要列出完整URL）

用自然、专业的中文回答。"""


def get_search_and_sql_prompt(question: str, search_results: str, sql_results: str) -> str:
    """联网搜索 + 数据库查询联合分析提示词
    
    Args:
        question: 用户问题
        search_results: 联网搜索结果
        sql_results: 数据库查询结果JSON
    
    Returns:
        联合分析提示词
    """
    return f"""你是一个数据分析专家，需要将行业外部数据（来自联网搜索）与公司内部数据（来自数据库）进行对比分析。

用户问题：{question}

【行业/外部数据（联网搜索）】
{search_results}

【公司内部数据（数据库查询）】
{sql_results}

请进行深度对比分析，包括：
1. **内外部数据概况**：分别简述两个数据来源的关键数字
2. **对比分析**：公司数据与行业数据的差距或优势
3. **亮点与问题**：公司在行业中的位置如何
4. **建议**：基于对比结果给出可操作的建议

请用结构化的方式呈现，突出对比结论。如果数据库查询结果为空或出错，请基于搜索结果给出通用分析。"""


def get_chart_config_prompt(data_summary: str, raw_data: str, context: str = "") -> str:
    """ECharts 图表配置生成提示词
    
    Args:
        data_summary: 数据摘要
        raw_data: 原始数据JSON字符串
        context: 上下文信息
    
    Returns:
        图表配置提示词
    """
    context_text = f"分析背景：{context}\n" if context else ""
    return f"""根据以下数据，生成一个适合可视化的 ECharts 图表配置对象（JSON格式）。

{context_text}数据摘要：
{data_summary}

原始数据：
{raw_data}

要求：
1. 选择最适合的图表类型（柱状图bar、折线图line、饼图pie）
2. 中文标题和标签
3. 只返回纯JSON对象，不要任何解释，不要代码块标记
4. 格式示例（柱状图）：
{{"title":{{"text":"标题"}},"tooltip":{{}},"xAxis":{{"data":["A","B"]}},"yAxis":{{}},"series":[{{"type":"bar","data":[1,2]}}]}}

注意：返回的必须是可以直接被JSON.parse()解析的合法JSON字符串。"""