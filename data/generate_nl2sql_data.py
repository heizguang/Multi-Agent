"""
NL2SQL 训练数据生成脚本

基于 employees, departments, salaries 三张表生成高质量的训练数据
"""

NL2SQL_TRAINING_DATA = [
    # ==================== 基础统计 ====================
    {
        "question": "公司一共有多少名员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees"
    },
    {
        "question": "公司有多少个部门？",
        "sql": "SELECT COUNT(*) AS department_count FROM departments"
    },
    {
        "question": "各部门分别有多少人？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name ORDER BY employee_count DESC"
    },
    {
        "question": "每个部门有多少员工？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "研发部有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "销售部有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '销售部'"
    },
    {
        "question": "市场部有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '市场部'"
    },
    {
        "question": "财务部有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '财务部'"
    },
    {
        "question": "人力资源部有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '人力资源部'"
    },
    {
        "question": "技术部有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '技术部'"
    },

    # ==================== 薪资统计 ====================
    {
        "question": "公司员工的平均薪资是多少？",
        "sql": "SELECT AVG(base_salary + bonus) AS avg_total_salary FROM salaries"
    },
    {
        "question": "平均工资最高的部门是哪个？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY avg_salary DESC LIMIT 1"
    },
    {
        "question": "哪个部门平均工资最高？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY avg_salary DESC LIMIT 1"
    },
    {
        "question": "平均工资最低的部门是哪个？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY avg_salary ASC LIMIT 1"
    },
    {
        "question": "各部门平均薪资是多少？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY avg_salary DESC"
    },
    {
        "question": "研发部平均工资是多少？",
        "sql": "SELECT AVG(s.base_salary + s.bonus) AS avg_salary FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "销售部平均薪资是多少？",
        "sql": "SELECT AVG(s.base_salary + s.bonus) AS avg_salary FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '销售部'"
    },
    {
        "question": "工资超过10000的员工有几个？",
        "sql": "SELECT COUNT(*) AS high_salary_count FROM salaries WHERE base_salary + bonus > 10000"
    },
    {
        "question": "工资超过20000的员工有多少？",
        "sql": "SELECT COUNT(*) AS employee_count FROM salaries WHERE base_salary + bonus > 20000"
    },
    {
        "question": "薪资最高的员工是谁？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 1"
    },
    {
        "question": "工资最高的员工是哪个部门的？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 1"
    },
    {
        "question": "工资最低的员工是谁？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary ASC LIMIT 1"
    },
    {
        "question": "薪资最高的前5名员工是谁？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 5"
    },
    {
        "question": "工资最高的前10名员工？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 10"
    },
    {
        "question": "研发部工资最高的3个人是谁？",
        "sql": "SELECT e.emp_name, e.position, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '研发部' ORDER BY total_salary DESC LIMIT 3"
    },
    {
        "question": "销售部业绩最好的员工是谁？",
        "sql": "SELECT e.emp_name, e.position, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '销售部' ORDER BY total_salary DESC LIMIT 1"
    },
    {
        "question": "公司总薪资支出是多少？",
        "sql": "SELECT SUM(base_salary + bonus) AS total_salary FROM salaries"
    },
    {
        "question": "各部门薪资总额是多少？",
        "sql": "SELECT d.dept_name, SUM(s.base_salary + s.bonus) AS total_salary FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY total_salary DESC"
    },
    {
        "question": "研发部总薪资是多少？",
        "sql": "SELECT SUM(s.base_salary + s.bonus) AS total_salary FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "基本工资超过15000的员工有哪些？",
        "sql": "SELECT e.emp_name, s.base_salary FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary > 15000"
    },
    {
        "question": "基本工资低于5000的员工有几个？",
        "sql": "SELECT COUNT(*) AS employee_count FROM salaries WHERE base_salary < 5000"
    },
    {
        "question": "奖金最高的人是谁？",
        "sql": "SELECT e.emp_name, s.bonus FROM employees e JOIN salaries s ON e.emp_id = s.emp_id ORDER BY s.bonus DESC LIMIT 1"
    },
    {
        "question": "没有奖金的员工有多少？",
        "sql": "SELECT COUNT(*) AS employee_count FROM salaries WHERE bonus = 0 OR bonus IS NULL"
    },
    {
        "question": "研发部平均基本工资是多少？",
        "sql": "SELECT AVG(s.base_salary) AS avg_base_salary FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "各部门的平均奖金是多少？",
        "sql": "SELECT d.dept_name, AVG(s.bonus) AS avg_bonus FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },

    # ==================== 员工信息 ====================
    {
        "question": "公司有多少名男性员工？",
        "sql": "SELECT COUNT(*) AS male_count FROM employees WHERE gender = '男'"
    },
    {
        "question": "公司有多少女性员工？",
        "sql": "SELECT COUNT(*) AS female_count FROM employees WHERE gender = '女'"
    },
    {
        "question": "各部门男女比例是多少？",
        "sql": "SELECT d.dept_name, SUM(CASE WHEN e.gender = '男' THEN 1 ELSE 0 END) AS male_count, SUM(CASE WHEN e.gender = '女' THEN 1 ELSE 0 END) AS female_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "研发部有多少男性？",
        "sql": "SELECT COUNT(*) AS male_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部' AND e.gender = '男'"
    },
    {
        "question": "销售部有多少女性员工？",
        "sql": "SELECT COUNT(*) AS female_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '销售部' AND e.gender = '女'"
    },
    {
        "question": "公司员工的男女比例是多少？",
        "sql": "SELECT gender, COUNT(*) AS count FROM employees GROUP BY gender"
    },
    {
        "question": "列出所有员工的名字和职位？",
        "sql": "SELECT emp_name, position FROM employees ORDER BY emp_name"
    },
    {
        "question": "公司有哪些职位？",
        "sql": "SELECT DISTINCT position FROM employees ORDER BY position"
    },
    {
        "question": "各部门的职位分布情况？",
        "sql": "SELECT d.dept_name, e.position, COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id GROUP BY d.dept_id, d.dept_name, e.position ORDER BY d.dept_name, employee_count DESC"
    },
    {
        "question": "每个职位有多少人？",
        "sql": "SELECT position, COUNT(*) AS employee_count FROM employees GROUP BY position ORDER BY employee_count DESC"
    },
    {
        "question": "研发部有哪些职位？",
        "sql": "SELECT DISTINCT e.position FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部' ORDER BY e.position"
    },
    {
        "question": "公司最常见的职位是什么？",
        "sql": "SELECT position, COUNT(*) AS employee_count FROM employees GROUP BY position ORDER BY employee_count DESC LIMIT 1"
    },

    # ==================== 入职时间 ====================
    {
        "question": "最早入职的员工是谁？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date ASC LIMIT 1"
    },
    {
        "question": "最近入职的员工是谁？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date DESC LIMIT 1"
    },
    {
        "question": "2023年入职的员工有多少？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees WHERE hire_date >= '2023-01-01' AND hire_date < '2024-01-01'"
    },
    {
        "question": "2024年新入职的员工有哪些？",
        "sql": "SELECT emp_name, hire_date, position FROM employees WHERE hire_date >= '2024-01-01' ORDER BY hire_date DESC"
    },
    {
        "question": "各部门平均入职时间？",
        "sql": "SELECT d.dept_name, AVG(julianday('now') - julianday(e.hire_date)) AS avg_days FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "工龄最长的员工是谁？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date ASC LIMIT 1"
    },
    {
        "question": "今年有多少员工入职？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees WHERE strftime('%Y', hire_date) = strftime('%Y', 'now')"
    },
    {
        "question": "上半年入职的员工数量？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees WHERE strftime('%m', hire_date) BETWEEN '01' AND '06'"
    },
    {
        "question": "按年份统计入职员工数量？",
        "sql": "SELECT strftime('%Y', hire_date) AS year, COUNT(*) AS employee_count FROM employees GROUP BY year ORDER BY year"
    },
    {
        "question": "哪个年份入职的员工最多？",
        "sql": "SELECT strftime('%Y', hire_date) AS year, COUNT(*) AS employee_count FROM employees GROUP BY year ORDER BY employee_count DESC LIMIT 1"
    },

    # ==================== 部门地点 ====================
    {
        "question": "各部门在哪个城市？",
        "sql": "SELECT dept_name, location FROM departments ORDER BY dept_name"
    },
    {
        "question": "公司在哪些城市有办公室？",
        "sql": "SELECT DISTINCT location FROM departments ORDER BY location"
    },
    {
        "question": "北京有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '北京'"
    },
    {
        "question": "上海有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '上海'"
    },
    {
        "question": "深圳有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '深圳'"
    },
    {
        "question": "广州有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '广州'"
    },
    {
        "question": "各城市分别有多少员工？",
        "sql": "SELECT d.location, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.location ORDER BY employee_count DESC"
    },
    {
        "question": "哪个城市员工最多？",
        "sql": "SELECT d.location, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.location ORDER BY employee_count DESC LIMIT 1"
    },
    {
        "question": "北京研发部有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '北京' AND d.dept_name = '研发部'"
    },
    {
        "question": "上海销售部有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.location = '上海' AND d.dept_name = '销售部'"
    },
    {
        "question": "各部门的办公地点在哪里？",
        "sql": "SELECT dept_name, location FROM departments ORDER BY dept_name"
    },

    # ==================== 复杂查询 ====================
    {
        "question": "研发部薪资最高的员工和最低的员工差距是多少？",
        "sql": "SELECT MAX(s.base_salary + s.bonus) - MIN(s.base_salary + s.bonus) AS salary_gap FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "各部门中薪资最高的人是谁？",
        "sql": "SELECT d.dept_name, e.emp_name, MAX(s.base_salary + s.bonus) AS max_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "平均薪资高于公司平均值的部门有哪些？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name HAVING avg_salary > (SELECT AVG(base_salary + bonus) FROM salaries)"
    },
    {
        "question": "薪资高于部门平均值的员工有哪些？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus > (SELECT AVG(s2.base_salary + s2.bonus) FROM salaries s2 JOIN employees e2 ON s2.emp_id = e2.emp_id WHERE e2.dept_id = e.dept_id)"
    },
    {
        "question": "没有部门的员工有多少？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees WHERE dept_id IS NULL"
    },
    {
        "question": "还没有工资信息的员工有哪些？",
        "sql": "SELECT e.emp_name, e.position FROM employees e LEFT JOIN salaries s ON e.emp_id = s.emp_id WHERE s.emp_id IS NULL"
    },
    {
        "question": "同时在多个部门工作过的员工有几个？",
        "sql": "SELECT emp_name, COUNT(DISTINCT dept_id) AS dept_count FROM employees GROUP BY emp_name HAVING dept_count > 1"
    },
    {
        "question": "各部门薪资排名前三的员工？",
        "sql": "SELECT d.dept_name, e.emp_name, s.base_salary + s.bonus AS total_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE (SELECT COUNT(*) FROM employees e2 JOIN salaries s2 ON e2.emp_id = s2.emp_id WHERE e2.dept_id = e.dept_id AND s2.base_salary + s2.bonus > s.base_salary + s.bonus) < 3 ORDER BY d.dept_name, total_salary DESC"
    },
    {
        "question": "工资最高和最低的部门差距有多大？",
        "sql": "SELECT MAX(avg_salary) - MIN(avg_salary) AS salary_gap FROM (SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id)"
    },
    {
        "question": "统计各部门的员工数量和平均薪资？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "按城市和部门统计员工人数？",
        "sql": "SELECT d.location, d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name, d.location ORDER BY d.location, employee_count DESC"
    },
    {
        "question": "工资超过部门平均值的员工比例是多少？",
        "sql": "SELECT ROUND(CAST(SUM(CASE WHEN total_salary > dept_avg THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS percentage FROM (SELECT e.emp_name, s.base_salary + s.bonus AS total_salary, (SELECT AVG(s2.base_salary + s2.bonus) FROM salaries s2 JOIN employees e2 ON s2.emp_id = e2.emp_id WHERE e2.dept_id = e.dept_id) AS dept_avg FROM employees e JOIN salaries s ON e.emp_id = s.emp_id)"
    },
    {
        "question": "找出工资异常低的员工（低于部门平均50%）？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus < (SELECT AVG(s2.base_salary + s2.bonus) * 0.5 FROM salaries s2 JOIN employees e2 ON s2.emp_id = e2.emp_id WHERE e2.dept_id = e.dept_id)"
    },
    {
        "question": "找出工资异常高的员工（高于部门平均200%）？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus > (SELECT AVG(s2.base_salary + s2.bonus) * 2 FROM salaries s2 JOIN employees e2 ON s2.emp_id = e2.emp_id WHERE e2.dept_id = e.dept_id)"
    },

    # ==================== 排序和筛选 ====================
    {
        "question": "按薪资从高到低排序所有员工？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC"
    },
    {
        "question": "按入职时间从早到晚排序？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date ASC"
    },
    {
        "question": "研发部员工按薪资排序？",
        "sql": "SELECT e.emp_name, e.position, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '研发部' ORDER BY total_salary DESC"
    },
    {
        "question": "找出工资在10000到20000之间的员工？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary, s.bonus, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus BETWEEN 10000 AND 20000"
    },
    {
        "question": "薪资在前20%的员工有多少？",
        "sql": "SELECT COUNT(*) AS top_20_percent FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus > (SELECT AVG(s2.base_salary + s2.bonus) FROM salaries s2) * 1.5"
    },
    {
        "question": "找出入职超过5年的员工？",
        "sql": "SELECT emp_name, hire_date, position FROM employees WHERE julianday('now') - julianday(hire_date) > 5 * 365"
    },
    {
        "question": "最近3年入职的员工中薪资最高的是谁？",
        "sql": "SELECT e.emp_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE e.hire_date >= date('now', '-3 years') ORDER BY total_salary DESC LIMIT 1"
    },

    # ==================== 高级统计 ====================
    {
        "question": "公司薪资的中位数是多少？",
        "sql": "SELECT AVG(base_salary + bonus) AS median_salary FROM (SELECT base_salary + bonus, ROW_NUMBER() OVER (ORDER BY base_salary + bonus) AS row_num, COUNT(*) OVER () AS total_count FROM salaries) WHERE row_num IN (total_count/2, total_count/2 + 1)"
    },
    {
        "question": "各部门薪资的中位数？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS median_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "薪资标准差最大的部门是哪个？",
        "sql": "SELECT d.dept_name, AVG((s.base_salary + s.bonus - dept_avg) * (s.base_salary + s.bonus - dept_avg)) AS variance FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id CROSS JOIN (SELECT AVG(s2.base_salary + s2.bonus) AS dept_avg FROM salaries s2 JOIN employees e2 ON s2.emp_id = e2.emp_id WHERE e2.dept_id = e.dept_id) GROUP BY d.dept_id, d.dept_name ORDER BY variance DESC LIMIT 1"
    },
    {
        "question": "员工薪资的分布情况？",
        "sql": "SELECT CASE WHEN base_salary + bonus < 5000 THEN '低' WHEN base_salary + bonus < 10000 THEN '中低' WHEN base_salary + bonus < 20000 THEN '中高' ELSE '高' END AS salary_level, COUNT(*) AS employee_count FROM salaries GROUP BY salary_level"
    },
    {
        "question": "各部门薪资占总薪资的比例？",
        "sql": "SELECT d.dept_name, ROUND(SUM(s.base_salary + s.bonus) * 100.0 / (SELECT SUM(base_salary + bonus) FROM salaries), 2) AS percentage FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY percentage DESC"
    },
    {
        "question": "基本工资占比最高的员工？",
        "sql": "SELECT e.emp_name, CAST(s.base_salary AS FLOAT) / (s.base_salary + s.bonus) AS base_ratio FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus > 0 ORDER BY base_ratio DESC LIMIT 1"
    },
    {
        "question": "奖金占比最高的员工？",
        "sql": "SELECT e.emp_name, CAST(s.bonus AS FLOAT) / (s.base_salary + s.bonus) AS bonus_ratio FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus > 0 ORDER BY bonus_ratio DESC LIMIT 1"
    },

    # ==================== 更多变体问题 ====================
    {
        "question": "告诉我公司有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees"
    },
    {
        "question": "查一下各部门的人数？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "研发部一共有多少员工？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "哪个部门人最多？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name ORDER BY employee_count DESC LIMIT 1"
    },
    {
        "question": "人最少的部门是哪个？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name ORDER BY employee_count ASC LIMIT 1"
    },
    {
        "question": "看一下研发部的平均工资？",
        "sql": "SELECT AVG(s.base_salary + s.bonus) AS avg_salary FROM salaries s JOIN employees e ON s.emp_id = e.emp_id JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "公司谁赚得最多？",
        "sql": "SELECT e.emp_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 1"
    },
    {
        "question": "收入最低的是谁？",
        "sql": "SELECT e.emp_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary ASC LIMIT 1"
    },
    {
        "question": "给我看看工资排名？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC"
    },
    {
        "question": "各部门都分布在哪？",
        "sql": "SELECT dept_name, location FROM departments"
    },
    {
        "question": "看看各城市的人员分布？",
        "sql": "SELECT d.location, COUNT(e.emp_id) AS employee_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.location"
    },
    {
        "question": "公司总共有多少个职位？",
        "sql": "SELECT COUNT(DISTINCT position) AS position_count FROM employees"
    },
    {
        "question": "我想知道每个部门有多少种职位？",
        "sql": "SELECT d.dept_name, COUNT(DISTINCT e.position) AS position_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "工作时间最长的员工？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date ASC LIMIT 1"
    },
    {
        "question": "新来的员工有哪些？",
        "sql": "SELECT emp_name, hire_date, position FROM employees ORDER BY hire_date DESC LIMIT 10"
    },
    {
        "question": "今年加入公司的员工有多少人？",
        "sql": "SELECT COUNT(*) AS employee_count FROM employees WHERE strftime('%Y', hire_date) = strftime('%Y', 'now')"
    },
    {
        "question": "男员工和女员工分别有多少？",
        "sql": "SELECT gender, COUNT(*) AS count FROM employees GROUP BY gender"
    },
    {
        "question": "各部门男女分别多少？",
        "sql": "SELECT d.dept_name, e.gender, COUNT(*) AS count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name, e.gender ORDER BY d.dept_name"
    },
    {
        "question": "研发部的同事们工资怎么样？",
        "sql": "SELECT e.emp_name, s.base_salary, s.bonus, s.base_salary + s.bonus AS total FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '研发部'"
    },
    {
        "question": "销售部谁最厉害？",
        "sql": "SELECT e.emp_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '销售部' ORDER BY total_salary DESC LIMIT 1"
    },
    {
        "question": "能不能帮我统计下薪资情况？",
        "sql": "SELECT MIN(base_salary + bonus) AS min_salary, MAX(base_salary + bonus) AS max_salary, AVG(base_salary + bonus) AS avg_salary FROM salaries"
    },
    {
        "question": "各部门的薪资水平如何？",
        "sql": "SELECT d.dept_name, MIN(s.base_salary + s.bonus) AS min_salary, MAX(s.base_salary + s.bonus) AS max_salary, AVG(s.base_salary + s.bonus) AS avg_salary FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "我想查询所有员工信息？",
        "sql": "SELECT e.emp_name, e.gender, e.hire_date, d.dept_name, e.position FROM employees e LEFT JOIN departments d ON e.dept_id = d.dept_id"
    },
    {
        "question": "帮我看看完整的员工和薪资信息？",
        "sql": "SELECT e.emp_name, e.gender, e.hire_date, d.dept_name, d.location, e.position, s.base_salary, s.bonus FROM employees e LEFT JOIN departments d ON e.dept_id = d.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id"
    },
    {
        "question": "哪个部门的开支最大？",
        "sql": "SELECT d.dept_name, SUM(s.base_salary + s.bonus) AS total_expense FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY total_expense DESC LIMIT 1"
    },
    {
        "question": "各部门平均基本工资是多少？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary) AS avg_base_salary FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "各部门的总奖金是多少？",
        "sql": "SELECT d.dept_name, SUM(s.bonus) AS total_bonus FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id LEFT JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY total_bonus DESC"
    },
    {
        "question": "找出薪资最高的那批人？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE s.base_salary + s.bonus >= (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY base_salary + bonus) FROM salaries)"
    },
    {
        "question": "收入在前10%的员工？",
        "sql": "SELECT e.emp_name, d.dept_name, s.base_salary + s.bonus AS total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT (SELECT CAST(COUNT(*) * 0.1 AS INTEGER) FROM employees)"
    },
    {
        "question": "我想了解下各部门的人员流动情况？",
        "sql": "SELECT d.dept_name, COUNT(e.emp_id) AS current_count, MIN(e.hire_date) AS earliest_hire, MAX(e.hire_date) AS latest_hire FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name"
    },
    {
        "question": "看看每个部门的薪资竞争力？",
        "sql": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary, (SELECT AVG(base_salary + bonus) FROM salaries) AS company_avg, AVG(s.base_salary + s.bonus) - (SELECT AVG(base_salary + bonus) FROM salaries) AS diff FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name"
    },
]


def export_to_jsonl(filepath: str = "data/nl2sql_train.jsonl"):
    """导出为 JSONL 格式（适合微调）"""
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in NL2SQL_TRAINING_DATA:
            # 格式化为 instruction-tuning 格式
            data = {
                "instruction": "根据用户问题生成SQL查询。只返回SQL语句，不要解释。",
                "input": f"问题：{item['question']}",
                "output": item['sql']
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"已导出 {len(NL2SQL_TRAINING_DATA)} 条数据到 {filepath}")


def export_to_alpaca(filepath: str = "data/nl2sql_alpaca.json"):
    """导出为 Alpaca 格式"""
    import json
    
    data = []
    for item in NL2SQL_TRAINING_DATA:
        data.append({
            "instruction": "根据用户问题生成SQL查询。只返回SQL语句，不要解释。",
            "input": f"问题：{item['question']}",
            "output": item['sql']
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已导出 {len(NL2SQL_TRAINING_DATA)} 条数据到 {filepath}")


if __name__ == "__main__":
    export_to_jsonl()
    export_to_alpaca()
    print(f"\n共生成 {len(NL2SQL_TRAINING_DATA)} 条训练数据")
