from collections.abc import Callable

import torch


def output_processor(strategy: str) -> Callable[[torch.Tensor], torch.Tensor]:
    match strategy:
        case "last":
            return lambda tensor: tensor[:, -1, :]

        case "concat":
            return lambda tensor: tensor.view(tensor.size(0), -1)

        case _:
            print(f"{strategy} is not supported")
