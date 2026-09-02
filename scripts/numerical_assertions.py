"""Cross-platform assertions for deterministic scientific rebuilds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from PIL import Image


def assert_nested_close(
    test_case: Any,
    actual: Any,
    expected: Any,
    *,
    path: str = "root",
    relative_tolerance: float = 2.0e-7,
    absolute_tolerance: float = 2.0e-9,
) -> None:
    """Compare nested records while allowing only small floating-point drift."""

    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        test_case.assertEqual(actual, expected, path)
        return
    if isinstance(expected, Integral):
        test_case.assertEqual(actual, expected, path)
        return
    if isinstance(expected, Real):
        test_case.assertIsInstance(actual, Real, path)
        test_case.assertTrue(
            math.isclose(
                float(actual),
                float(expected),
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ),
            f"{path}: {actual!r} != {expected!r}",
        )
        return
    if isinstance(expected, Mapping):
        test_case.assertIsInstance(actual, Mapping, path)
        test_case.assertEqual(set(actual), set(expected), path)
        for key in expected:
            assert_nested_close(
                test_case,
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, Sequence):
        test_case.assertIsInstance(actual, Sequence, path)
        test_case.assertEqual(len(actual), len(expected), path)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_nested_close(
                test_case,
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    test_case.assertEqual(actual, expected, path)


def assert_raster_equivalent(test_case: Any, rebuilt: Path, committed: Path) -> None:
    """Verify cross-platform figure structure without freezing font metrics."""

    test_case.assertTrue(rebuilt.is_file(), rebuilt)
    test_case.assertGreater(rebuilt.stat().st_size, 1500, rebuilt)
    with Image.open(rebuilt) as rebuilt_image, Image.open(committed) as committed_image:
        test_case.assertEqual(rebuilt_image.mode, committed_image.mode)
        test_case.assertGreaterEqual(rebuilt_image.width, 1500)
        test_case.assertGreaterEqual(rebuilt_image.height, 1000)
        rebuilt_aspect = rebuilt_image.width / rebuilt_image.height
        committed_aspect = committed_image.width / committed_image.height
        test_case.assertGreater(rebuilt_aspect / committed_aspect, 0.70)
        test_case.assertLess(rebuilt_aspect / committed_aspect, 1.40)
        extrema = rebuilt_image.convert("L").getextrema()
        test_case.assertIsNotNone(extrema)
        test_case.assertLess(extrema[0], extrema[1])
