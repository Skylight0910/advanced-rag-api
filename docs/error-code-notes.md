# 故障排查记录

系统出现错误码 ZXQ-9471 时，原因通常是索引写入完成后尚未执行 flush 和 load_collection。

处理方式：
1. 执行 client.flush(collection_name=COLLECTION_NAME)
2. 执行 client.load_collection(collection_name=COLLECTION_NAME)
3. 再次发起检索请求
