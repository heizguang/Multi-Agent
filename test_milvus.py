from pymilvus import connections, utility
import sys

# 配置连接信息
# 根据你的截图，端口映射是 19530:19530，所以 host 是 localhost，port 是 19530
HOST = "localhost"
PORT = "19530"

def test_connection():
    print(f"正在尝试连接到 Milvus 服务器 {HOST}:{PORT} ...")

    try:
        # 1. 建立连接
        # alias="default" 是默认的连接别名
        connections.connect(alias="default", host=HOST, port=PORT)

        # 2. 检查连接状态
        # get_connection_addr 会返回连接地址信息，如果未连接则返回空
        addr = connections.get_connection_addr("default")
        print(f"连接地址配置: {addr}")

        # 3. 尝试获取服务器版本
        # 如果服务器未就绪，这一步通常会超时或报错
        version = utility.get_server_version()
        print(f"成功连接到 Milvus！")
        print(f"服务器版本: {version}")

        # 4. (可选) 列出所有集合，进一步验证读写权限
        collections = utility.list_collections()
        print(f"当前数据库中的集合数量: {len(collections)}")
        # print(f"集合列表: {collections}")

    except Exception as e:
        print(f"连接失败！错误信息: {e}")
        sys.exit(1)
    finally:
        # 断开连接
        connections.disconnect("default")

if __name__ == "__main__":
    test_connection()
