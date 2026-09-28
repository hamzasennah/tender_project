from dataclasses import dataclass


@dataclass(frozen=True)
class PromptEngineeringContext:
    text: str
    metadata: dict
