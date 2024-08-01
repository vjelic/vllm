from torch import nn

from vllm.config import MultiModalConfig


class VisionLanguageModelBase(nn.Module):
    """Base class for all vision language models (VLMs)."""

    def __init__(self, multimodal_config: MultiModalConfig) -> None:
        super().__init__()

        self.multimodal_config = multimodal_config
