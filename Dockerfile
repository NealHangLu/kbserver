FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
EXPOSE 5601
CMD ["python3", "app.py"]