import unittest
import itertools
import copy
import torch
import torch.nn.functional as F
from torch.optim import SGD
import intel_extension_for_pytorch as ipex

from common_utils import TestCase

conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
convtranspose_module = {
    1: torch.nn.ConvTranspose1d,
    2: torch.nn.ConvTranspose2d,
    3: torch.nn.ConvTranspose3d,
}


class ConvNd_Relu(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super(ConvNd_Relu, self).__init__()
        self.conv = conv_module[dim](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class Linear_Relu(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Linear_Relu, self).__init__()
        self.linear = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return F.relu(self.linear(x))


class DeconvNd_Relu(torch.nn.Module):
    def __init__(
        self,
        dim,
        ic,
        oc,
        kernel_size,
    ):
        super(DeconvNd_Relu, self).__init__()
        self.deconv = convtranspose_module[dim](
            ic,
            oc,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return F.relu(self.deconv(x))


class Lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


class BmmAdd(torch.nn.Module):
    def __init__(self):
        super(BmmAdd, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        res = torch.add(bmm_res, input)
        return res


class TestCompileCases(TestCase):
    def test_conv_relu_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["torchscript", "inductor"],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
            ) in options:
                if compiler_backend == "torchscript" and dynamic is True:
                    continue
                if weight_prepack is True and ipex_optimize is False:
                    continue
                N = 2
                M = 2
                C = 3
                x_shape = (N, C) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = ConvNd_Relu(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                ).eval()
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    model = ipex.optimize(
                        model, weights_prepack=weight_prepack, dtype=dtype
                    )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                    for _ in range(3):
                        y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_conv_relu_train(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["inductor"],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
            ) in options:
                if weight_prepack is True and ipex_optimize is False:
                    continue
                N = 2
                M = 2
                C = 3
                x_shape = (N, C) + input_shapes[dim]
                input = torch.randn(x_shape, dtype=torch.float32)
                ori_x = input.clone().requires_grad_()
                x = input.clone().requires_grad_()
                conv = ConvNd_Relu(
                    dim=dim,
                    in_channels=C,
                    out_channels=M,
                    kernel_size=3,
                )
                ori_model = copy.deepcopy(conv).train()
                model = copy.deepcopy(conv).train()
                optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    ori_model, _ = ipex.optimize(
                        ori_model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
                    model, _ = ipex.optimize(
                        model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ):
                    ori_y = ori_model(ori_x)
                    y = compile_model(x)
                    grad_x = torch.randn(y.shape, dtype=torch.float32)
                    ori_y.backward(grad_x)
                    y.backward(grad_x)
                    self.assertEqual(y, ori_y)
                    self.assertTrue(y.dtype == dtype)
                    self.assertEqual(x.grad, ori_x.grad)

    def test_deconv_relu_inference(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            input_channel_per_group = 6
            output_channel_per_group = 3
            kernel_size = 3
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["torchscript", "inductor"],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
            ) in options:
                if compiler_backend == "torchscript" and dynamic is True:
                    continue
                if weight_prepack is True and ipex_optimize is False:
                    continue
                ic = input_channel_per_group
                oc = output_channel_per_group
                x_shape = (2, ic) + input_shapes[dim]
                x = torch.randn(x_shape, dtype=torch.float32)
                model = DeconvNd_Relu(dim, ic, oc, kernel_size).eval()
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    model = ipex.optimize(
                        model, weights_prepack=weight_prepack, dtype=dtype
                    )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ), torch.no_grad():
                    ori_y = model(x)
                    for _ in range(3):
                        y = compile_model(x)
                self.assertEqual(y, ori_y)
                self.assertTrue(y.dtype == dtype)

    def test_deconv_relu_train(self):
        for dim in [1, 2, 3]:
            input_shapes = {1: (4,), 2: (4, 4), 3: (4, 4, 4)}
            input_channel_per_group = 6
            output_channel_per_group = 3
            kernel_size = 3
            options = itertools.product(
                [torch.float32, torch.bfloat16],
                ["inductor"],
                [True, False],
                [True, False],
                [True, False],
            )
            for (
                dtype,
                compiler_backend,
                dynamic,
                ipex_optimize,
                weight_prepack,
            ) in options:
                if weight_prepack is True and ipex_optimize is False:
                    continue
                ic = input_channel_per_group
                oc = output_channel_per_group
                x_shape = (2, ic) + input_shapes[dim]
                input = torch.randn(x_shape, dtype=torch.float32)
                ori_x = input.clone().requires_grad_()
                x = input.clone().requires_grad_()
                deconv = DeconvNd_Relu(dim, ic, oc, kernel_size)
                ori_model = copy.deepcopy(deconv).train()
                model = copy.deepcopy(deconv).train()
                optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
                if ipex_optimize:
                    # TODO: support channels_last_1d.
                    if dim == 1:
                        ipex.disable_auto_channels_last()
                    else:
                        ipex.enable_auto_channels_last()
                    ori_model, _ = ipex.optimize(
                        ori_model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
                    model, _ = ipex.optimize(
                        model,
                        weights_prepack=weight_prepack,
                        dtype=dtype,
                        optimizer=optimizer,
                    )
                torch._dynamo.reset()
                ipex._set_compiler_backend(compiler_backend)
                compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
                with torch.cpu.amp.autocast(
                    enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
                ):
                    ori_y = ori_model(ori_x)
                    y = compile_model(x)
                    grad_x = torch.randn(ori_y.shape, dtype=torch.float32)
                    ori_y.backward(grad_x)
                    y.backward(grad_x)
                    self.assertEqual(y, ori_y)
                    self.assertTrue(y.dtype == dtype)
                    self.assertEqual(x.grad, ori_x.grad)

    def test_linear_relu_inference(self):
        out_features = 4
        in_features = 3
        input_shapes = [(2, in_features), (2, 2, in_features), (2, 2, 2, in_features)]
        options = itertools.product(
            input_shapes,
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            x_shape,
            dtype,
            compiler_backend,
            dynamic,
            ipex_optimize,
            weight_prepack,
        ) in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            if weight_prepack is True and ipex_optimize is False:
                continue
            x = torch.randn(x_shape, dtype=torch.float32)
            model = Linear_Relu(in_features, out_features).eval()
            if ipex_optimize:
                model = ipex.optimize(
                    model, weights_prepack=weight_prepack, dtype=dtype
                )
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x)
                for _ in range(3):
                    y = compile_model(x)
            self.assertEqual(y, ori_y, prec=0.01)
            self.assertTrue(y.dtype == dtype)

    def test_linear_relu_train(self):
        out_features = 4
        in_features = 3

        input_shapes = [(2, in_features), (2, 2, in_features), (2, 2, 2, in_features)]
        options = itertools.product(
            input_shapes,
            [torch.float32, torch.bfloat16],
            ["inductor"],
            [True, False],
            [True, False],
            [True, False],
        )
        for (
            x_shape,
            dtype,
            compiler_backend,
            dynamic,
            ipex_optimize,
            weight_prepack,
        ) in options:
            if weight_prepack is True and ipex_optimize is False:
                continue
            input = torch.randn(x_shape, dtype=torch.float32)
            ori_x = input.clone().requires_grad_()
            x = input.clone().requires_grad_()
            linear = Linear_Relu(in_features, out_features)
            ori_model = copy.deepcopy(linear).train()
            model = copy.deepcopy(linear).train()
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
            if ipex_optimize:
                ori_model, _ = ipex.optimize(
                    ori_model,
                    weights_prepack=weight_prepack,
                    dtype=dtype,
                    optimizer=optimizer,
                )
                model, _ = ipex.optimize(
                    model,
                    weights_prepack=weight_prepack,
                    dtype=dtype,
                    optimizer=optimizer,
                )
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ):
                ori_y = ori_model(ori_x)
                y = compile_model(x)
                grad_x = torch.randn(ori_y.shape, dtype=torch.float32)
                ori_y.backward(grad_x)
                y.backward(grad_x)
                self.assertEqual(y, ori_y, prec=0.01)
                self.assertTrue(y.dtype == dtype)
                self.assertEqual(x.grad, ori_x.grad, prec=0.01)

    def test_lstm_inference(self):
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
            [True, False],
        )
        for dtype, compiler_backend, dynamic, ipex_optimize in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            input = torch.randn(5, 3, 10)
            h0 = torch.randn(2, 3, 20)
            c0 = torch.randn(2, 3, 20)
            model = Lstm(10, 20, 2).eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_output, (ori_hn, ori_cn) = model(input, (h0, c0))
            if ipex_optimize:
                model = ipex.optimize(model, dtype=dtype)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                output, (hn, cn) = compile_model(input, (h0, c0))
            self.assertEqual(ori_output, output)
            self.assertEqual(ori_hn, hn)
            self.assertEqual(ori_cn, cn)
            self.assertTrue(output.dtype == dtype)
            self.assertTrue(hn.dtype == dtype)
            self.assertTrue(cn.dtype == dtype)

    def test_bmm_add_inference(self):
        options = itertools.product(
            [torch.float32, torch.bfloat16],
            ["torchscript", "inductor"],
            [True, False],
        )
        for dtype, compiler_backend, dynamic in options:
            if compiler_backend == "torchscript" and dynamic is True:
                continue
            x = torch.randn(6, 3, 5).to(dtype=dtype)
            b1 = torch.randn(6, 3, 4).to(dtype=dtype)
            b2 = torch.randn(6, 4, 5).to(dtype=dtype)
            model = BmmAdd().eval()
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                ori_y = model(x, b1, b2)
            torch._dynamo.reset()
            ipex._set_compiler_backend(compiler_backend)
            compile_model = torch.compile(model, dynamic=dynamic, backend="ipex")
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y = compile_model(x, b1, b2)
            self.assertEqual(ori_y, y, prec=0.1)
            self.assertTrue(y.dtype == dtype)


if __name__ == "__main__":
    torch.manual_seed(2020)
    test = unittest.main()
