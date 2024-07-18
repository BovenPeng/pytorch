# mypy: allow-untyped-defs
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
from .optimizer import (
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _foreach_doc,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

__all__ = ["Adagrad", "adagrad"]


class Adagrad(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            fused_supported_devices = _get_fused_kernels_supported_devices()
            # Not support CUDA yet
            fused_supported_devices.remove("cuda")
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

        # Q. 怎么理解存的state
        # A. sum是历史梯度的平方和
         in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # 某一版实现是 `step = torch.tensor(0.0)`, 用浮点数保持类型一致
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    # Q. API torch.is_complex
                    # A. p是否为复数, 常用于信号(声音和图像)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                # Q. API torch.full_like, torch.preserve_format
                # A. 创建一个和指定 Tensor 的形状 (shape)，数据类型 (dtype) 以及硬件设备 (device, 如 CPU 或 GPU) 相同的新 Tensor，并且新 Tensor 的每个元素都被设置为指定的值; memory_format指内存格式(sparse, contiguous)
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        #  define "fused" for
        #  MYPY error: Name "fused" may be undefined
        fused = None
         in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
        
        # 确保 "step" 键的值是 Tensor 类型。如果不是 Tensor 类型，它将在后面部分代码中被转化为 Tensor
        state_values = list(self.state.values())
        # Q. API torch.is_tensor()
        # A. 用于检查某个对象是否是 PyTorch Tensor 对象。
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(
                    float(s["step"]), dtype=_get_scalar_dtype(is_fused=fused)
                )
    # 用于在多个进程之间共享模型参数，这在多进程训练时非常有用。开启多进程训练可以更好地利用硬件资源并加速训练过程。由于每个进程都有自己的内存空间，所以默认情况下，各进程中的变量值不共享
    def share_memory(self):
         in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Q. API .share_memory_()
                # A. Tensor 提供了 share_memory_() 这个方法，用于共享在不同进程之间的内存。
                state["sum"].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        has_sparse_grad, has_complex = False, False
        for p in group["params"]:
            if p.grad is not None:
                # Q. API p.grad.is_sparse
                # A. 检查参数 p 的梯度是否是稀疏的。如果 p 的梯度是稀疏的，那么 has_sparse_grad 就被设置为 True。然后，将参数 p，它的梯度，以及和 p 相关联的状态加入到相应的列表中。在后面的优化步骤中，这些列表将会被用于更新 p 的值。
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        return has_sparse_grad, has_complex

    # Q. @_use_grad_for_differentiable
    # A. @_use_grad_for_differentiable 装饰器可能用于确保 step 函数只更新那些可微分的参数。
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            # Q. API param `use_fused`, data is on device or host
            with torch.enable_grad():
                loss = closure()

         in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            state_sums: List[Tensor] = []
            state_steps: List[Tensor] = []

            has_sparse_grad, has_complex = self._init_group(
                group, params_with_grad, grads, state_sums, state_steps
            )

            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


Adagrad.__doc__ = (
    r"""Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow \tau                          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the
            sum of squares of gradients (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        fused (bool, optional): whether the fused implementation (CPU only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None). Please note that the fused implementations does not
            support sparse or complex gradients.
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html

    """
)


def adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adagrad
    # Q. 为什么要分 multi和single tensor adagrad呢?
    # A. 1. 处理多个/单个tensor; 2. 并行计算是否开启(参数foreach); 3. 是否在TorchScript模式下运行
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad

    func(
        params,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _make_sparse(grad, grad_indices, values):
    """
    # 稀疏 tensor 的 indices
    indices = torch.tensor([[0, 1, 1],
                            [2, 0, 2]])

    # 稀疏 tensor 的 values
    values = torch.tensor([3, 4, 5])

    # 用 indices 和 values 构造 grad
    grad = torch.sparse_coo_tensor(indices, values, [2, 4])

    # 使用函数
    sparse_tensor = _make_sparse(grad, indices, values)

    # 打印输出
    print(sparse_tensor.to_dense())

    tensor([[0, 0, 3, 0],
        [4, 0, 5, 0]])
    """
    size = grad.size()
    return torch.sparse_coo_tensor(grad_indices, values, size)


def _single_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None
    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        # update step
        step_t += 1
        # Q. 为什么 _get_value(step_t) ?
        # A. 使用 _get_value 函数从 step_t（这是一个 Tensor 对象）中取出值，并将它赋值给 step
        #   x.item() 返回一个标准 Python 数字，这是一个标量值
        #   x 返回一个包含该值的 Tensor
        step = _get_value(step_t)
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            # grad += param * weight_decay
            grad = grad.add(param, alpha=weight_decay)

        # (step - 1) * lr_decay作为decay
        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            # Q. API coalesce(), _indices(), _values(), _make_sparse(), sparse_mask()
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()

            # _make_sparse() 根据这些索引和平方值更新历史平方梯度和
            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            # 获取 state_sum 在 grad 非零元素位置的值，计算出每个位置的 std 值（即历史平方梯度和的平方根）
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
            )
        else:
            is_complex = torch.is_complex(param)

            # switch complex num 2 real num
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            # Q. API addcmul_
            # element-wise相乘并且就地累加: state_sum += value * (grad * grad)
            state_sum.addcmul_(grad, grad, value=1)
            # 可微分 <=> 是否要反向传播
            if differentiable:
                std = state_sum.sqrt() + eps
            else:
                std = state_sum.sqrt().add_(eps)

            # Q. API addcdiv_(), view_as_complex()
            # element-wise相除并且地累加: param += (-clr) * (grad / std)
            param.addcdiv_(grad, std, value=-clr)
            # swith real num back complex num
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)


def _multi_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert not differentiable, "_foreach ops don't support autograd"
    assert grad_scale is None and found_inf is None

    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:
        return
        
    # 将 参数params, 梯度grads, 历史梯度平方和state_sums, 步数steps 按照device和dtype来划分; 因为数据是 GPU and CPU fused
    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )

    # 将同一device, dtype的tensor进行并行计算
    for (
        device_params,
        device_grads,
        device_state_sums,
        device_state_steps,
    ), _ in grouped_tensorlists.values():

        # 同一组内只要有一个参数的梯度为sparse, 则调用_single_tensor_adagrad来进行优化
        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if device_has_sparse_grad:
            _single_tensor_adagrad(
                device_params,
                device_grads,
                device_state_sums,
                device_state_steps,
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                eps=eps,
                has_sparse_grad=True,
                maximize=maximize,
                differentiable=differentiable,
                has_complex=has_complex,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )
            continue

        # Handle complex parameters
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)

        # 优化目标最大化 <=> 优化目标最小化 和 梯度相反
        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                # device_grads = device_grad + device_param * alpha
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        minus_clr = [
            -lr / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps
        ]

        # each ele, device_state_sums += value * (device_grads * device_grads)
        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)

        # each ele, device_state_sums = sqrt(device_state_sums)
        std = torch._foreach_sqrt(device_state_sums)
        # each ele, std += eps
        torch._foreach_add_(std, eps)

        if weight_decay != 0 or maximize:
            # Again, re-use the intermediate memory (device_grads) already allocated
            torch._foreach_mul_(device_grads, minus_clr)
            numerator = device_grads
        else:
            numerator = torch._foreach_mul(device_grads, minus_clr)  # type: ignore[assignment]

        # device_params += 1 * std/ numerator
        torch._foreach_addcdiv_(device_params, numerator, std)


def _fused_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad or has_complex:
        raise RuntimeError("`fused` does not support sparse grad or complex param")

    if differentiable:
        raise RuntimeError(
            "adagrad with fused=True does not support differentiable=True"
        )

    grad_scale_dict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else None
    )
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )
    for (device, _), (
        (
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
        ),
        _,
    ) in grouped_tensors.items():
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None and grad_scale_dict is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)  # type: ignore[index]
            device_grad_scale = grad_scale_dict[device]  # type: ignore[index]
        if found_inf is not None and found_inf_dict is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)  # type: ignore[index]
            device_found_inf = found_inf_dict[device]  # type: ignore[index]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adagrad_(
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )
