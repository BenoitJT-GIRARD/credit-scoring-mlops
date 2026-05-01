FROM prom/prometheus:latest

COPY docker/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
