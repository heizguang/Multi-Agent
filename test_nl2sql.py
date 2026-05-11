"""
测试微调后的 NL2SQL 模型
"""

import os
import sys
import torch
import re

# 模型路径
MODEL_PATH = "./models/nl2sql-qwen3.5-4b/final"

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
    test_model()
