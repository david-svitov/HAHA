import torch
import torch.nn as nn


@torch.no_grad()
def cat_tensors_to_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    N = -1
    for group in optimizer.param_groups:
        if group["name"] not in tensors_dict.keys():
            # print(f"Warning: {group['name']} not in optimizer, skip")
            continue
        assert len(group["params"]) == 1, f"{group['name']} has more than one param"
        extension_tensor = tensors_dict[group["name"]]
        # print(group["name"])
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"].clone(), torch.zeros_like(extension_tensor)),
                dim=0,
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (
                    stored_state["exp_avg_sq"].clone(),
                    torch.zeros_like(extension_tensor),
                ),
                dim=0,
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=0,
                )
                .contiguous()
                .requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=0,
                ).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def update_tensors_in_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] not in tensors_dict.keys():
            # print(f"Warning: {group['name']} not in optimizer, skip")
            continue
        assert len(group["params"]) == 1, f"{group['name']} has more than one param"
        extension_tensor = tensors_dict[group["name"]]

        stored_state = optimizer.state.get(group["params"][0], None)
        stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)
        del optimizer.state[group["params"][0]]

        group["params"][0] = nn.Parameter(extension_tensor.contiguous()).requires_grad_(True)
        optimizer.state[group["params"][0]] = stored_state
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def prune_optimizer(optimizer, mask, exclude_names=[]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        # print(group["name"])
        if group["name"] in exclude_names or len(group["params"]) == 0:
            continue
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                (group["params"][0][mask].requires_grad_(True))
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][mask].requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors
