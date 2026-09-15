FROM prom/prometheus:v3.14.0

COPY infra/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
