# The data source and the dashboard are baked into the image rather than bind-mounted, for
# the same reason as the Prometheus configuration: a container that depends on a host path
# only starts correctly on the host that has it.
FROM grafana/grafana:13.2.1

COPY infra/grafana/provisioning /etc/grafana/provisioning
COPY infra/grafana/dashboards /etc/grafana/dashboards
