FROM python:3.7.3-slim
COPY . /web
WORKDIR /web/api
RUN pip install -r ./requirements.txt
RUN useradd -ms /bin/bash chuan
USER chuan
ENTRYPOINT ["python"]
CMD ["app.py"]