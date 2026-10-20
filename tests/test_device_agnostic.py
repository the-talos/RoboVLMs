"""Device selection and dispatch tests for the inference path (no hardware needed)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from robovlms.utils import device as dev
from robovlms.model.policy_head.base_policy import LSTMDecoder
from robovlms.train.base_trainer import BaseTrainer


def fake_host(xpu=False, cuda=False):
    available = {"xpu": xpu, "cuda": cuda}
    return mock.patch.object(
        dev, "is_available", side_effect=lambda t: available.get(t, t == "cpu")
    )


class DeviceSelectionTest(unittest.TestCase):
    def test_default_type_prefers_xpu_then_cuda_then_cpu(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
        with fake_host(cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")

    def test_resolve_device(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device(), torch.device("xpu", 0))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda"), torch.device("cuda", 0))
            self.assertEqual(dev.resolve_device("xpu:1"), torch.device("xpu", 1))
        with fake_host():
            self.assertEqual(dev.resolve_device(), torch.device("cpu"))

    def test_dist_backend(self):
        self.assertEqual(dev.dist_backend("cuda"), "nccl")
        self.assertEqual(dev.dist_backend("xpu"), "xccl")
        self.assertEqual(dev.dist_backend("cpu"), "gloo")
        with fake_host(xpu=True):
            self.assertEqual(dev.dist_backend(), "xccl")

    def test_set_device_dispatch(self):
        with mock.patch.object(torch.cuda, "set_device") as cuda_set, mock.patch.object(
            torch, "xpu", create=True
        ) as xpu:
            dev.set_device(torch.device("cuda", 1))
            cuda_set.assert_called_once_with(torch.device("cuda", 1))
            dev.set_device(torch.device("xpu", 2))
            xpu.set_device.assert_called_once_with(torch.device("xpu", 2))
            with fake_host(xpu=True):
                dev.set_device(3)
            xpu.set_device.assert_called_with(torch.device("xpu", 3))
            dev.set_device("cpu")
            self.assertEqual(cuda_set.call_count, 1)
            self.assertEqual(xpu.set_device.call_count, 2)


class ProcessBatchTest(unittest.TestCase):
    def test_batch_follows_module_device_not_cuda(self):
        stub = SimpleNamespace(
            device=torch.device("cpu"),
            use_hand_rgb=True,
            configs={"window_size": 2},
        )
        batch = {
            "rgb": torch.zeros(1, 2, 3, 8, 8),
            "hand_rgb": torch.zeros(1, 2, 3, 8, 8),
            "text": torch.ones(1, 4, dtype=torch.long),
            "text_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        out = BaseTrainer._process_batch(stub, batch)
        self.assertEqual(out[0].device.type, "cpu")
        self.assertEqual(out[1].device.type, "cpu")
        self.assertEqual(out[3].shape, (1, 4))


def tiny_head():
    torch.manual_seed(0)
    head = LSTMDecoder(
        in_features=16,
        action_dim=7,
        down_sample="none",
        latent=1,
        fwd_pred_next_n=3,
        window_size=4,
        hidden_size=16,
        num_layers=2,
    ).eval()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return head


@unittest.skipUnless(dev.is_available("xpu"), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_lstm_head_matches_cpu(self):
        cpu_head = tiny_head()
        xpu_head = tiny_head().to("xpu")
        tok = torch.randn(2, 4, 1, 16)
        with torch.no_grad():
            ref_arm, ref_grip = cpu_head(tok)
            arm, grip = xpu_head(tok.to("xpu"))
        self.assertGreater(ref_arm.std().item(), 1e-3)
        torch.testing.assert_close(arm.cpu(), ref_arm, atol=2e-3, rtol=1e-3)
        torch.testing.assert_close(grip.cpu(), ref_grip, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
