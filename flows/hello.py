from prefect import flow

@flow(name="hello-world")
def hello_flow(name: str = "World") -> None:
    print(f"Hello, {name}!")


if __name__ == "__main__":
    hello_flow()
