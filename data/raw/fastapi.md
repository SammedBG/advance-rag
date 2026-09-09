# FastAPI Architecture and Best Practices

## Dependency Injection

FastAPI includes a powerful Dependency Injection system that allows developers to declare dependencies in path operations.

### Using Depends

Dependencies are declared using `Depends()`. Common use cases include:
- Database session management
- Authentication and authorization token verification
- Shared business logic services
- Request validation and rate limiting

## Async Endpoints and Background Tasks

### Async vs Sync Path Operations

FastAPI supports both standard `def` functions and asynchronous `async def` endpoints:
- Use `async def` when performing non-blocking I/O operations such as querying an async database or making external HTTP requests.
- Use `def` for CPU-intensive tasks or synchronous blocking libraries, which FastAPI automatically executes in an external threadpool.

### Background Tasks

Background tasks allow operations to run after returning an HTTP response.
Common examples include sending email notifications, processing analytical logs, and updating search indexes.
