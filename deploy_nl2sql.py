"""
NL2SQL 模型部署脚本
将微调后的模型集成到 SQL Agent 中
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 微调模型路径
MODEL_PATH = os.environ.get("NL2SQL_MODEL_PATH", "./models/nl2sql-qwen3.5-4b/final")

# 全局变量
nl2sql_pipeline = None


def load_model():
    """加载微调后的 NL2SQL 模型"""
    global nl2sql_pipeline
    
    if not os.path.exists(MODEL_PATH):
        print(f"[警告] 模型路径不存在: {MODEL_PATH}")
        print("[信息] 将使用原始 LLM")
        return None
    
    print(f"[信息] 加载微调模型: {MODEL_PATH}")
    
    try:
        # 加载 tokenizer 和模型
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型（量化版）
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 创建 pipeline
        nl2sql_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("[成功] 微调模型加载完成")
        return nl2sql_pipeline
        
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return None


def generate_sql(question: str, schema: str = "") -> str:
    """使用微调模型生成 SQL
    
    Args:
        question: 用户问题
        schema: 数据库结构描述（可选）
    
    Returns:
        生成的 SQL 语句
    """
    global nl2sql_pipeline
    
    # 如果 pipeline 未加载，尝试加载
    if nl2sql_pipeline is None:
        load_model()
    
    # 构建 prompt
    if schema:
        prompt = f"""根据用户问题生成SQL查询。只返回SQL语句。

数据库结构：
{schema}

问题：{question}
SQL："""
    else:
        prompt = f"""根据用户问题生成SQL查询。只返回SQL语句。

问题：{question}
SQL："""
    
    try:
        if nl2sql_pipeline is not None:
            # 使用微调模型
            result = nl2sql_pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                return_full_text=False
            )
            output = result[0]["generated_text"]
        else:
            # 回退到原始 LLM
            print("[信息] 使用原始 LLM")
            from llm_client import OpenAICompatRequestsLLM
            
            llm = OpenAICompatRequestsLLM(
                model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                temperature=0.1,
                max_tokens=256
            )
            
            response = llm.invoke(prompt)
            output = response.content if hasattr(response, 'content') else str(response)
        
        # 提取 SQL
        sql = extract_sql(output)
        return sql
        
    except Exception as e:
        print(f"[错误] SQL 生成失败: {e}")
        return ""


def extract_sql(text: str) -> str:
    """从模型输出中提取 SQL"""
    import re
    
    text = text.strip()
    
    # 移除 markdown 代码块
    text = re.sub(r'^```sql\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text)
    
    # 移除常见前缀
    prefixes = ['SQL：', 'SQL:', 'sql:', 'sql：']
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    # 清理空白
    text = text.strip().rstrip(';').strip()
    
    return text


def replace_sql_agent():
    """替换 SQL Agent 中的 LLM 调用"""
    import sys
    sys.path.insert(0, '.')
    
    from agents.sql_agent import SQLQueryAgent
    
    # 保存原始方法
    original_generate_sql = SQLQueryAgent._generate_sql
    
    def new_generate_sql(self, question: str) -> str:
        """使用微调模型生成 SQL"""
        # 获取 schema
        schema = self._get_schema(question=question)
        
        # 使用微调模型
        sql = generate_sql(question, schema)
        
        # 如果失败，回退到原始方法
        if not sql:
            print("[信息] 回退到原始 LLM")
            return original_generate_sql(self, question)
        
        return self._clean_sql(sql)
    
    # 替换方法
    SQLQueryAgent._generate_sql = new_generate_sql
    print("[成功] SQL Agent 已替换为微调模型")


def start_server():
    """启动 API 服务"""
    from flask import Flask, request, jsonify
    import threading
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "model_loaded": nl2sql_pipeline is not None})
    
    @app.route('/generate_sql', methods=['POST'])
    def gen_sql():
        data = request.json
        question = data.get('question', '')
        schema = data.get('schema', '')
        
        sql = generate_sql(question, schema)
        
        return jsonify({
            "question": question,
            "sql": sql,
            "success": bool(sql)
        })
    
    print("[信息] 启动 API 服务: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NL2SQL 模型部署')
    parser.add_argument('--mode', choices=['load', 'replace', 'server'], default='load',
                       help='运行模式: load=加载模型, replace=替换SQL Agent, server=启动API')
    parser.add_argument('--port', type=int, default=8080, help='API服务端口')
    args = parser.parse_args()
    
    if args.mode == 'load':
        # 仅加载模型
        load_model()
        
    elif args.mode == 'replace':
        # 替换 SQL Agent
        load_model()
        replace_sql_agent()
        print("[信息] 可以启动主应用了")
        
    elif args.mode == 'server':
        # 启动 API 服务
        load_model()
        start_server()


if __name__ == "__main__":
    main()
