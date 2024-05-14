import torch

from apex.contrib.layer_norm.layer_norm import FastRMSNormFN, FastLayerNormFN
from faster_norm import faster_rms_norm, faster_layer_norm


torch.manual_seed(42)

s, b, h = 8192, 1, 12288

hidden_states = (4. + torch.randn(s, b, h, dtype=torch.bfloat16, device="cuda")).requires_grad_()
weight = (1 + torch.rand(h, dtype=torch.bfloat16, device="cuda")).requires_grad_()
bias = (1 + torch.rand(h, dtype=torch.bfloat16, device="cuda")).requires_grad_()
eps = 1e-5
grad_output = torch.randn(s, b, h, dtype=torch.bfloat16, device="cuda") + hidden_states.detach() * .01


def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    if hidden_states.dtype != torch.float64:
        hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def layer_norm(hidden_states, weight, bias, variance_epsilon):
    return torch.nn.functional.layer_norm(hidden_states, weight.shape, weight, bias, variance_epsilon)


rms_norm_compiled = torch.compile(rms_norm)
layer_norm_compiled = torch.compile(layer_norm)


def fast_rms_norm(x, gamma, epsilon):
    return FastRMSNormFN.apply(x, gamma, epsilon)


def fast_layer_norm(x, gamma, beta, epsilon):
    return FastLayerNormFN.apply(x, gamma, beta, epsilon)


for is_layer_norm in [False, True]:
    dtype_ref = torch.float64
    hidden_states_float = hidden_states.detach().to(dtype_ref).requires_grad_()
    weight_float = weight.detach().to(dtype_ref).requires_grad_()
    bias_float = bias.detach().to(dtype_ref).requires_grad_()
    if is_layer_norm:
        output_ref = layer_norm(hidden_states_float, weight_float, bias_float, eps).to(hidden_states.dtype)
    else:
        output_ref = rms_norm(hidden_states_float, weight_float, eps).to(hidden_states.dtype)
    output_ref.backward(grad_output)
    dgrad_ref = hidden_states_float.grad.to(hidden_states.dtype)
    wgrad_ref = weight_float.grad.to(weight.dtype)
    if is_layer_norm:
        bgrad_ref = bias_float.grad.to(bias.dtype)

    for do_backward in ["", "[input]", "[input]+[weight]+[bias]"]:
        weight.requires_grad_("[weight]" in do_backward)
        bias.requires_grad_("[bias]" in do_backward)
        for _ in range(3):
            for name, fn in ([
                ("layer_norm_native", layer_norm),
                ("layer_norm_compiled", layer_norm_compiled),
                ("FastLayerNormFN", fast_layer_norm),
                ("faster_layer_norm", faster_layer_norm),
            ] if is_layer_norm else [
                ("rms_norm_native", rms_norm),
                ("rms_norm_compiled", rms_norm_compiled),
                ("FastRMSNormFN", fast_rms_norm),
                ("faster_rms_norm", faster_rms_norm),
            ]):
                warmup_times = 5
                run_times = 20
                ev1 = torch.cuda.Event(enable_timing=True)
                ev2 = torch.cuda.Event(enable_timing=True)
                for _ in range(warmup_times):
                    hidden_states.grad = None
                    weight.grad = None
                    bias.grad = None
                    if is_layer_norm:
                        output = fn(hidden_states, weight, bias, eps)
                    else:
                        output = fn(hidden_states, weight, eps)
                    if do_backward:
                        output.backward(grad_output)
                ev1.record()
                for _ in range(run_times):
                    hidden_states.grad = None
                    weight.grad = None
                    bias.grad = None
                    if is_layer_norm:
                        output = fn(hidden_states, weight, bias, eps)
                    else:
                        output = fn(hidden_states, weight, eps)
                    if do_backward:
                        output.backward(grad_output)
                ev2.record()
                ev2.synchronize()
                dt = ev1.elapsed_time(ev2) / 1000. / run_times
                bw = hidden_states.numel() * hidden_states.element_size() * (5 if do_backward else 2) / dt
                print(f"{name:20s}", f"{bw / 2**30:8.3f} GiB/s", end=" ")
                print(f"{(output - output_ref).abs().mean().item():.3E}", end=" ")
                if do_backward:
                    print(f"{(hidden_states.grad - dgrad_ref).abs().mean().item():.3E}", end=" ")
                    if weight.requires_grad:
                        print(f"{(weight.grad - wgrad_ref).abs().mean().item():.3E}", end=" ")
                    if is_layer_norm and bias.requires_grad:
                        print(f"{(bias.grad - bgrad_ref).abs().mean().item():.3E}", end=" ")
                print()
