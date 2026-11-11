# kb_qa 生产镜像：Python 3.11 精简版
FROM python:3.11-slim

WORKDIR /app

# 先只拷依赖清单再安装：代码改动后重新构建时，依赖层走缓存，不用重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再拷项目代码（.dockerignore 里排除的东西不会进来）
COPY . .

EXPOSE 8000

# uvicorn 直接加载 main.py 里的 app（Gradio 已挂载在 /ui）
# --host 0.0.0.0 是容器里必须的：只监听 127.0.0.1 的话外面访问不到
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
