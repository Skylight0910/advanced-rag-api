# Advanced RAG API 故障排查手册

## Ubuntu 无法通过 Windows FIClash 访问外网

### 症状
Ubuntu 虚拟机无法直接使用宿主机代理访问外网。

### 原因
FIClash 通常只监听 Windows 宿主机的 `127.0.0.1:7890`。
Ubuntu 虚拟机无法直接访问宿主机 loopback 地址。

### 解决方法
使用以下链路：

Ubuntu -> 192.168.31.1:7891 -> Windows portproxy -> 127.0.0.1:7890 -> FIClash

Ubuntu 中设置：

```bash
export http_proxy=http://192.168.31.1:7891
export https_proxy=http://192.168.31.1:7891
export HTTP_PROXY=http://192.168.31.1:7891
export HTTPS_PROXY=http://192.168.31.1:7891
export NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0
export no_proxy=localhost,127.0.0.1,::1,0.0.0.0
如果 Windows 重启后代理失效，在 Windows 检查：

netstat -ano | findstr 7890
netstat -ano | findstr 7891
netsh interface portproxy show all
Get-Service iphlpsvc
访问 localhost API 时返回 502 Bad Gateway
症状
curl http://localhost:8000/docs
返回：

HTTP/1.1 502 Bad Gateway
原因
curl 将 localhost 请求也发送给了代理。

解决方法
export NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0
export no_proxy=localhost,127.0.0.1,::1,0.0.0.0
临时绕过代理测试：

curl --noproxy localhost,127.0.0.1 http://localhost:8000/
缺少 socksio
症状
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed
原因
环境中设置了 all_proxy 或 ALL_PROXY，程序尝试使用 SOCKS 代理。

解决方法
unset all_proxy
unset ALL_PROXY
保留 HTTP 和 HTTPS 代理即可。

docker compose 命令不可用
症状
docker compose
无法使用。

原因
当前 Ubuntu 环境没有 Compose Plugin。

解决方法
使用旧版独立命令：

docker-compose
当前环境无需重复尝试安装 docker-compose-plugin。

Docker Hub 拉取镜像失败
症状
connect: connection refused
connection reset by peer
原因
即使为 dockerd 配置了代理，Docker Hub 连接仍可能失败。

解决方法
使用已验证可用的 DaoCloud 镜像前缀：

sudo docker pull docker.m.daocloud.io/minio/minio:RELEASE.2024-12-18T13-15-44Z
sudo docker pull docker.m.daocloud.io/milvusdb/milvus:v2.6.17
Milvus 插入成功但搜索结果为空
症状
数据已经插入 Milvus，但检索不到结果。

原因
插入后没有及时 flush 和 load collection。

解决方法
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(collection_name=COLLECTION_NAME)
API 启动时无法连接 Milvus
症状
Fail connecting to server on localhost:19530
原因
app.py 启动时会初始化 Milvus，但 Milvus 容器尚未启动。

解决方法
cd ~/code/p1-project/milvus
sudo docker-compose up -d
sudo docker-compose ps
查看日志：

sudo docker-compose logs standalone
/ask 接口返回请求字段错误
症状
请求体使用了：

{"question": "..."}
原因
/ask 接口要求字段名为 query。

解决方法
{"query": "..."}
Milvus 启动时重复重建 collection
症状
每次启动 API 都会删除并重建 Milvus collection，导致重复 Embedding 和启动缓慢。

原因
REBUILD_MILVUS_ON_START = True
解决方法
首次迁移完成后修改为：

REBUILD_MILVUS_ON_START = False
uvicorn --reload 导致模型重复加载
症状
Embedding 或 Reranker 模型可能重复加载，也更容易遇到 HuggingFace 网络和缓存问题。

原因
--reload 会启动额外进程并监视文件变化。

解决方法
当前开发阶段优先使用：

uvicorn app:app --host 0.0.0.0 --port 8000
