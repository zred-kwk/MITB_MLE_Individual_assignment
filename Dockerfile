# Shared Jupyter and Airflow image.
FROM apache/airflow:2.10.5

ENV DEBIAN_FRONTEND=noninteractive

# Install Java for PySpark and common shell tools.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
