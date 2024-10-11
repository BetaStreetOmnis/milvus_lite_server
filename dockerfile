# 使用官方的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到工作目录
COPY . /app

# 安装依赖，使用清华源
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 设置容器最大空间为1G
RUN echo "Storage=1G" >> /etc/systemd/system.conf

# 暴露FastAPI默认端口
EXPOSE 8089

# # 挂载当前目录的db文件
# VOLUME ["/app/db"]

# 设置容器自动重启
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8089/ || exit 1
LABEL com.centurylinklabs.watchtower.enable=true

# 运行FastAPI应用
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8089"]


# docker run -d -p 8089:8089 --name milvus_lite_api5 -v /root/milvus_lite_server/db:/app/db my_milvus_lite