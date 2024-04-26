from anthropic_iterative_search import final_chain

if __name__ == "__main__":
    query = (
        "How do you run a job? or "
        "How do I do this or this in 'software app'"
    )
    print(  # noqa: T201
        final_chain.with_config(configurable={"chain": "retrieve"}).invoke(
            {"query": query}
        )
    )
