"""Minimal Prefect flow that can be deployed from a Git repository."""

from datetime import datetime

from prefect import flow, task


@task(name="compose_message")
def compose_message(name: str) -> str:
    """Build a friendly greeting with a timestamp."""
    return f"Hello, {name}! The current UTC time is {datetime.utcnow().isoformat()}"


@task(name="emit_message")
def emit_message(message: str) -> None:
    """Print the message; Prefect captures stdout in the run logs."""
    print(message)


@flow(name="git-prefect-sample")
def git_prefect_sample(name: str = "Prefect User") -> None:
    """Simple flow to demonstrate Git + Prefect deployments."""
    message = compose_message(name)
    emit_message(message)


if __name__ == "__main__":
    git_prefect_sample()
