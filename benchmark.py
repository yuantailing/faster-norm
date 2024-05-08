import torch

from apex.contrib.layer_norm.layer_norm import FastRMSNormFN
from fast_norm_cuda import fast_rms_norm_cuda


torch.manual_seed(42)

s, b, h = 8192, 1, 12288

hidden_states = torch.randn(s, b, h, dtype=torch.bfloat16, device="cuda").requires_grad_()
weight = (1 + torch.rand(h, dtype=torch.bfloat16, device="cuda")).requires_grad_()
eps = 1e-5
grad_output = torch.randn(s, b, h, dtype=torch.bfloat16, device="cuda") + hidden_states.detach() * .01


def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    if hidden_states.dtype != torch.float64:
        hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


rms_norm_compiled = torch.compile(rms_norm)


def fast_rms_norm(x, gamma, epsilon):
    return FastRMSNormFN.apply(x, gamma, epsilon)


dtype_ref = torch.float64
hidden_states_float = hidden_states.detach().to(dtype_ref).requires_grad_()
weight_float = weight.detach().to(dtype_ref).requires_grad_()
output_ref = rms_norm(hidden_states_float, weight_float, eps).to(hidden_states.dtype)
output_ref.backward(grad_output)
dgrad_ref = hidden_states_float.grad.to(hidden_states.dtype)
wgrad_ref = weight_float.grad.to(weight.dtype)


for do_backward in [False, True]:
    for _ in range(5):
        for name, fn in [
            ("rms_norm_native", rms_norm),
            ("rms_norm_compiled", rms_norm_compiled),
            ("FastRMSNormFN", fast_rms_norm),
            ("fast_rms_norm_cuda", fast_rms_norm_cuda),
        ]:
            warmup_times = 5
            run_times = 20
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            for _ in range(warmup_times):
                hidden_states.grad = None
                weight.grad = None
                output = fn(hidden_states, weight, eps)
                if do_backward:
                    output.backward(grad_output)
            ev1.record()
            for _ in range(run_times):
                hidden_states.grad = None
                weight.grad = None
                output = fn(hidden_states, weight, eps)
                if do_backward:
                    output.backward(grad_output)
            ev2.record()
            ev2.synchronize()
            dt = ev1.elapsed_time(ev2) / 1000. / run_times
            bw = hidden_states.numel() * hidden_states.element_size() * (2 + do_backward * 3) / dt
            print(f"{name:20s}", f"{bw / 2**30:8.3f} GiB/s", end=" ")
            print((output - output_ref).abs().mean().item(), end=" ")
            if do_backward:
                print((hidden_states.grad - dgrad_ref).abs().mean().item(), end=" ")
                print((weight.grad - wgrad_ref).abs().mean().item(), end=" ")
            print()
