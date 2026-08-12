from minivllm import EngineConfig, LLMEngine, SamplingParams


def main() -> None:
    config = EngineConfig(data_parallel_size=2, max_num_batched_tokens=64)
    with LLMEngine(config) as engine:
        for output in engine.generate(
            "Explain paged attention:", SamplingParams(max_tokens=10)
        ):
            print(output.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
