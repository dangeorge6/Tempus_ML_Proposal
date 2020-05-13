FROM jupyter/pyspark-notebook

RUN pip install --upgrade pip && \
    pip install --no-cache-dir pandas pyarrow matplotlib pyspark koalas

COPY koalas_demo.py /usr/local/bin
COPY breast_cancer_data.csv /usr/local/bin

CMD ["spark-submit", "/usr/local/bin/koalas_demo.py"]