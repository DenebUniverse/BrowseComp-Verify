# Synced from the server; standalone demo blocks omitted.
from dataclasses import dataclass, field
from typing import Optional, Dict

Message = dict[str, str]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError