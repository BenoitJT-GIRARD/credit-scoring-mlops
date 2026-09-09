FROM prom/prometheus:v3.14.0

COPY docker/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
