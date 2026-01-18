#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest
from functools import partial
from typing import Any

import torch

from noether.core.distributed import (
    all_gather_grad,
    all_gather_nograd,
    all_gather_nograd_clipped,
    all_reduce_mean_grad,
    all_reduce_sum_grad,
)


class TestGatherNondistributed(unittest.TestCase):
    @staticmethod
    def _all_ops():
        return [
            # (op, is_gather, has_grad)
            (all_gather_grad, True, True),
            (all_gather_nograd, True, False),
            (partial(all_gather_nograd_clipped, max_length=None), True, False),
            (all_reduce_sum_grad, False, True),
            (all_reduce_mean_grad, False, True),
        ]

    def test_floatscalar(self):
        source = 5.0
        for op, is_gather, _ in self._all_ops():
            x = op(source)
            self.assertTrue(torch.is_tensor(x))
            if is_gather:
                self.assertEqual((1,), x.shape)
            else:
                self.assertEqual(0, x.ndim)
            self.assertEqual(torch.float32, x.dtype)
            self.assertEqual(5.0, x)
            self.assertIsNone(x.grad_fn)

    def test_intscalar(self):
        source = 5
        for op, is_gather, _ in self._all_ops():
            x = op(source)
            self.assertTrue(torch.is_tensor(x))
            if is_gather:
                self.assertEqual((1,), x.shape)
            else:
                self.assertEqual(0, x.ndim)
            self.assertEqual(torch.int64, x.dtype)
            self.assertEqual(5, x)
            self.assertIsNone(x.grad_fn)

    def test_floatlist(self):
        source: tuple[Any, ...] = (5.0, 6.0)
        for op, _, _ in self._all_ops():
            for to_tuple in [False, True]:
                if to_tuple:
                    source = tuple(source)
                x = op(source)
                self.assertTrue(torch.is_tensor(x))
                self.assertEqual((2,), x.shape)
                self.assertEqual(torch.float32, x.dtype)
                self.assertEqual(5.0, x[0])
                self.assertEqual(6.0, x[1])
                self.assertIsNone(x.grad_fn)

    def test_intlist(self):
        source = (5, 6)
        for op, _, _ in self._all_ops():
            for to_tuple in [False, True]:
                if to_tuple:
                    source = tuple(source)  # type: ignore
                x = op(source)
                self.assertTrue(torch.is_tensor(x))
                self.assertEqual((2,), x.shape)
                self.assertEqual(torch.int64, x.dtype)
                self.assertEqual(5, x[0])
                self.assertEqual(6, x[1])
                self.assertIsNone(x.grad_fn)

    def test_tensor(self):
        source = torch.randn(5, 6)
        for op, _, has_grad in self._all_ops():
            source_clone = source.clone()
            source_clone.requires_grad = True
            # * 5 to make a computation graph
            source_with_graph = source_clone * 5
            x = op(source_with_graph)
            self.assertTrue(torch.is_tensor(x))
            self.assertEqual((5, 6), x.shape)
            self.assertEqual(torch.float32, x.dtype)
            self.assertTrue(torch.all(source_with_graph == x))
            self.assertEqual(has_grad, x.grad_fn is not None)
            if has_grad:
                x.sum().backward()
                self.assertTrue(torch.all(torch.full_like(source, fill_value=5.0) == source_clone.grad))
            else:
                with self.assertRaises(RuntimeError) as ex:
                    x.sum().backward()
                msg = "element 0 of tensors does not require grad and does not have a grad_fn"
                self.assertEqual(msg, str(ex.exception))
