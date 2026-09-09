# Docker Containerization Guide

## Multi-Stage Builds

Multi-stage builds allow you to drastically reduce final container image size by separating the build environment from the runtime environment.

### Optimization Best Practices

Key best practices for production Docker images include:
- Use slim base images such as `python:3.10-slim` or `alpine`
- Order instructions from least frequently changing to most frequently changing to maximize layer cache hits
- Always combine `apt-get update` and `apt-get install` in a single `RUN` layer and clean `/var/lib/apt/lists/*`
- Run containers as non-root users for enhanced security

## Container Health Checks

Health checks ensure the container orchestrator knows when a service is healthy and ready to accept traffic.
Use `HEALTHCHECK` with custom intervals, timeout periods, and retry thresholds.
