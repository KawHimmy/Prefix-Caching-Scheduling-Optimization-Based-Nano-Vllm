from nanovllm.sampling_params import SamplingParams


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLM", "SamplingParams"]
