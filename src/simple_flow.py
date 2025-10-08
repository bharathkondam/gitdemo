from prefect import flow

@flow
def hello_flow(name: str = "Prefect") -> None:
    print(f"Hello, {name}!")


if __name__ == "__main__":
    hello_flow()
