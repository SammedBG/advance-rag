# Kubernetes Troubleshooting

## Pod Restarting

A Kubernetes pod may restart when its container exits unexpectedly.

### CrashLoopBackOff

CrashLoopBackOff means Kubernetes is repeatedly starting and restarting
a container.

Common causes include:

- Application crashes
- Missing environment variables
- Failed health checks
- Insufficient resources

## ImagePullBackOff

ImagePullBackOff occurs when Kubernetes cannot pull the container image.

Check:

- Image name
- Image tag
- Registry credentials
- Network connectivity