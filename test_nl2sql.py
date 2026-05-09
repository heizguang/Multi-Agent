"""
测试微调后的 NL2SQL 模型

支持两种模式：
1) local: 直接加载本地模型目录进行测试
2) vllm: 通过 OpenAI 兼容接口测试（自动读取 /v1/models）
"""

import argparse
import os
import re
import torch
import requests

MODEL_PATH = "./models/nl2sql-qwen3.5-4b/final"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:6006")


def resolve_model_name(base_url: str, model_name: str = "") -> str:
    """优先使用传入模型名，否则从 /v1/models 自动获取。"""
    if model_name:
        return model_name

    url = f"{base_url.rstrip('/')}/v1/models"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    models = data.get("data", [])
    if not models:
        raise RuntimeError("未从 /v1/models 获取到模型列表")

    first = models[0]
    resolved = first.get("id") if isinstance(first, dict) else ""
    if not resolved:
        raise RuntimeError("模型列表格式异常，无法解析 model id")
    return resolved

def test_model():
    print("=" * 50)
    print("测试微调后的 NL2SQL 模型")
    print("=" * 50)
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型路径不存在: {MODEL_PATH}")
        return
    
    print(f"[信息] 加载模型: {MODEL_PATH}")
    
    # 导入必要的库
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import re
    
    # 加载 tokenizer
    print("[信息] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("[信息] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print("[成功] 模型加载完成\n")
    
    # 测试问题
    test_questions = [
        "公司一共有多少名员工？",
        "各部门分别有多少人？",
        "研发部平均工资是多少？",
        "工资最高的员工是谁？",
        "各部门的平均薪资是多少？",
    ]
    
    # Schema 信息
    schema = """
表：employees
  - emp_id: INTEGER (主键)
  - emp_name: TEXT
  - gender: TEXT
  - hire_date: TEXT
  - dept_id: INTEGER
  - position: TEXT

表：departments
  - dept_id: INTEGER (主键)
  - dept_name: TEXT
  - location: TEXT

表：salaries
  - emp_id: INTEGER (主键)
  - base_salary: REAL
  - bonus: REAL
"""
    
    print("=" * 50)
    print("开始测试")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] 问题: {question}")
        
        # 构建 prompt
        prompt = f"""根据用户问题生成SQL查询。只返回SQL语句。

数据库结构：
{schema}

问题：{question}
SQL："""
        
        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取 SQL
        sql = extract_sql(output)
        
        print(f"    SQL: {sql}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


def test_vllm(base_url: str, model_name: str = ""):
    print("=" * 50)
    print("测试 vLLM NL2SQL 接口")
    print("=" * 50)

    resolved_model = resolve_model_name(base_url, model_name)
    print(f"[信息] 服务地址: {base_url}")
    print(f"[信息] 使用模型: {resolved_model}")

    test_questions = [
        "公司一共有多少名员工？",
        "各部门分别有多少人？",
        "研发部平均工资是多少？",
        "工资最高的员工是谁？",
        "各部门的平均薪资是多少？",
    ]

    schema = """
表：employees
  - emp_id: INTEGER (主键)
  - emp_name: TEXT
  - gender: TEXT
  - hire_date: TEXT
  - dept_id: INTEGER
  - position: TEXT

表：departments
  - dept_id: INTEGER (主键)
  - dept_name: TEXT
  - location: TEXT

表：salaries
  - emp_id: INTEGER (主键)
  - base_salary: REAL
  - bonus: REAL
"""

    url = f"{base_url.rstrip('/')}/v1/completions"
    print("=" * 50)
    print("开始测试")
    print("=" * 50)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] 问题: {question}")
        prompt = f"""根据用户问题生成SQL查询。只返回SQL语句。

数据库结构：
{schema}

问题：{question}
SQL："""
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.1,
            "stop": ["<|im_end|>", "\n\n"],
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            output = resp.json()["choices"][0]["text"]
            sql = extract_sql(output)
            print(f"    SQL: {sql}")
        except Exception as e:
            print(f"    [错误] 请求失败: {e}")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


def extract_sql(text: str) -> str:
    """从模型输出中提取 SQL"""
    # 查找最后一个 "SQL：" 后的内容
    # 尝试找到最后的 SQL 部分
    match = re.search(r'SQL[：:]\s*(.+?)(?:(?:\n\n)|(?:\n[^S])|$)', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        # 回退：取最后一行
        lines = text.strip().split('\n')
        sql = lines[-1].strip()
    
    # 移除 markdown 代码块
    sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'^```\s*', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'```$', '', sql)
    
    # 清理空白
    sql = sql.strip().rstrip(';').strip()
    
    return sql


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 NL2SQL 模型（local 或 vllm）")
    parser.add_argument("--mode", choices=["local", "vllm"], default="vllm")
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help="vLLM 服务地址")
    parser.add_argument("--model", default="", help="可选，指定模型名；不传则自动从 /v1/models 获取")
    args = parser.parse_args()

    if args.mode == "local":
        test_model()
    else:
        test_vllm(args.base_url, args.model)
