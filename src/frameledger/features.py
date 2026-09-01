from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TableChangeGeometry:
    """Spatial support for a visual change inside the worksheet body."""

    changed_ratio: float
    row_coverage: float
    column_coverage: float
    highlight_changed_ratio: float
    highlight_row_coverage: float
    highlight_column_coverage: float
    residual_changed_ratio: float
    residual_row_coverage: float
    residual_column_coverage: float


@dataclass(frozen=True)
class PresentationChangeGeometry:
    """Spatial evidence for a change inside the useful slide canvas.

    ``edge_loss_ratio`` is normalized by the ROI area, while
    ``retained_edge_ratio`` is normalized by the number of edges in the older
    state.  Keeping both makes an additive build distinguishable from a page
    replacement without treating a sparse slide as if it contained no visual
    structure.
    """

    changed_ratio: float
    soft_changed_ratio: float
    edge_changed_ratio: float
    block_spread: float
    retained_edge_ratio: float
    edge_loss_ratio: float
    edge_gain_ratio: float
    left_edge_ratio: float
    right_edge_ratio: float
    largest_component_ratio: float
    largest_component_width_ratio: float
    largest_component_height_ratio: float
    component_count: int

    @property
    def old_edge_retention(self) -> float:
        """Alias spelling the direction of ``retained_edge_ratio``."""
        return self.retained_edge_ratio

    @property
    def old_edge_loss_ratio(self) -> float:
        """Alias spelling the direction of ``edge_loss_ratio``."""
        return self.edge_loss_ratio


@dataclass(frozen=True)
class RoutingFrameFeatures:
    """OCR-free layout signals measured inside the main content body."""

    mean_luma: float
    dark_ratio: float
    bright_ratio: float
    edge_density: float
    horizontal_line_ratio: float
    vertical_line_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_luma": round(self.mean_luma, 6),
            "dark_ratio": round(self.dark_ratio, 6),
            "bright_ratio": round(self.bright_ratio, 6),
            "edge_density": round(self.edge_density, 6),
            "horizontal_line_ratio": round(self.horizontal_line_ratio, 6),
            "vertical_line_ratio": round(self.vertical_line_ratio, 6),
        }


def resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("Analysis width must be positive")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] <= width:
        return gray
    height = max(1, int(round(gray.shape[0] * width / gray.shape[1])))
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def routing_frame_features(
    gray: np.ndarray,
    *,
    top_ratio: float = 0.08,
    bottom_ratio: float = 0.90,
    left_ratio: float = 0.02,
    right_ratio: float = 0.98,
) -> RoutingFrameFeatures:
    """Measure conservative content-routing signals in a grayscale frame.

    The morphology lengths are calibrated for a 480-pixel analysis frame and
    scale with the body ROI width.  Opening the Canny mask keeps long worksheet
    grid lines while rejecting most glyph strokes and pointer edges.
    """
    if gray.ndim != 2:
        raise ValueError("Routing features require a grayscale image")
    if not 0 <= top_ratio < bottom_ratio <= 1:
        raise ValueError("Routing vertical ROI must satisfy 0 <= top < bottom <= 1")
    if not 0 <= left_ratio < right_ratio <= 1:
        raise ValueError("Routing horizontal ROI must satisfy 0 <= left < right <= 1")

    height, width = gray.shape
    x0 = min(width - 1, int(round(width * left_ratio)))
    x1 = max(x0 + 1, min(width, int(round(width * right_ratio))))
    y0 = min(height - 1, int(round(height * top_ratio)))
    y1 = max(y0 + 1, min(height, int(round(height * bottom_ratio))))
    body = gray[y0:y1, x0:x1]
    edges = cv2.Canny(body, 80, 160)

    morphology_length = max(3, int(round(24 * body.shape[1] / 480.0)))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (morphology_length, 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, morphology_length),
    )
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

    return RoutingFrameFeatures(
        mean_luma=float(body.mean()),
        dark_ratio=float(np.mean(body < 64)),
        bright_ratio=float(np.mean(body > 220)),
        edge_density=float(np.mean(edges > 0)),
        horizontal_line_ratio=float(np.mean(horizontal_lines > 0)),
        vertical_line_ratio=float(np.mean(vertical_lines > 0)),
    )


def perceptual_hash(gray: np.ndarray) -> int:
    """Return a conventional 64-bit DCT perceptual hash."""
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].copy()
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def phash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("SSIM inputs must have the same shape")
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    covariance = float(((a - mean_a) * (b - mean_b)).mean())
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def pixel_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Pixel-delta inputs must have the same shape")
    return float(cv2.absdiff(left, right).mean() / 255.0)


def edge_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Edge-delta inputs must have the same shape")
    left_edges = cv2.Canny(left, 80, 160) > 0
    right_edges = cv2.Canny(right, 80, 160) > 0
    return float(np.logical_xor(left_edges, right_edges).mean())


def block_change(left: np.ndarray, right: np.ndarray, grid: int = 8) -> float:
    """Measure local change so a small edited cell is not hidden by global similarity."""
    if left.shape != right.shape:
        raise ValueError("Block-change inputs must have the same shape")
    diff = cv2.absdiff(left, right).astype(np.float32) / 255.0
    height, width = diff.shape
    block_scores: list[float] = []
    for row in range(grid):
        y0 = row * height // grid
        y1 = (row + 1) * height // grid
        for column in range(grid):
            x0 = column * width // grid
            x1 = (column + 1) * width // grid
            block_scores.append(float(diff[y0:y1, x0:x1].mean()))
    # The maximum block protects a small spreadsheet cell or chart annotation
    # that would disappear in a whole-frame mean. Other similarity gates keep
    # isolated compression noise from causing a keep by itself.
    return max(block_scores, default=0.0)


def worksheet_change_geometry(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_ratio: float = 0.25,
    bottom_ratio: float = 0.92,
    pixel_threshold: int = 12,
    active_row_ratio: float = 0.04,
    active_column_ratio: float = 0.02,
    wide_row_ratio: float = 0.60,
    maximum_highlight_band_ratio: float = 0.20,
    bridge_rows: int = 2,
    highlight_padding_rows: int = 1,
) -> TableChangeGeometry:
    """Describe table changes after separating narrow full-width row bands.

    The representative videos devote the centre of the frame to a worksheet.
    A selected company or row changes a thin horizontal band across most
    columns, while a scroll, sort, or sheet switch leaves change spread through
    the worksheet body.  This function measures both signals without depending
    on Excel chrome, text recognition, or a particular workbook skin.
    """
    if left.shape != right.shape:
        raise ValueError("Worksheet-change inputs must have the same shape")
    if left.ndim != 2:
        raise ValueError("Worksheet-change inputs must be grayscale images")
    if not 0 <= top_ratio < bottom_ratio <= 1:
        raise ValueError("Worksheet ROI ratios must satisfy 0 <= top < bottom <= 1")
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("Pixel threshold must be between 0 and 255")
    for name, value in (
        ("active row ratio", active_row_ratio),
        ("active column ratio", active_column_ratio),
        ("wide row ratio", wide_row_ratio),
        ("maximum highlight band ratio", maximum_highlight_band_ratio),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name.capitalize()} must be between 0 and 1")
    if bridge_rows < 0 or highlight_padding_rows < 0:
        raise ValueError("Highlight bridge and padding rows must be non-negative")

    height = left.shape[0]
    y0 = min(height - 1, int(round(height * top_ratio)))
    y1 = max(y0 + 1, min(height, int(round(height * bottom_ratio))))
    mask = cv2.absdiff(left[y0:y1], right[y0:y1]) >= pixel_threshold
    roi_height = mask.shape[0]

    row_density = mask.mean(axis=1)
    wide_rows = np.flatnonzero(row_density >= wide_row_ratio)
    highlight_rows = np.zeros(roi_height, dtype=bool)
    if wide_rows.size:
        band_start = int(wide_rows[0])
        band_end = band_start
        bands: list[tuple[int, int]] = []
        for row in wide_rows[1:]:
            row = int(row)
            if row - band_end <= bridge_rows + 1:
                band_end = row
            else:
                bands.append((band_start, band_end))
                band_start = band_end = row
        bands.append((band_start, band_end))
        maximum_band_rows = max(1, int(round(roi_height * maximum_highlight_band_ratio)))
        for band_start, band_end in bands:
            # A broad contiguous change is a viewport event, not a selection
            # highlight, and must remain in the residual mask.
            if band_end - band_start + 1 > maximum_band_rows:
                continue
            start = max(0, band_start - highlight_padding_rows)
            end = min(roi_height, band_end + highlight_padding_rows + 1)
            highlight_rows[start:end] = True

    highlight_mask = np.zeros_like(mask)
    highlight_mask[highlight_rows] = mask[highlight_rows]
    residual_mask = mask.copy()
    residual_mask[highlight_rows] = False

    def coverage(change_mask: np.ndarray) -> tuple[float, float, float]:
        return (
            float(change_mask.mean()),
            float(np.mean(change_mask.mean(axis=1) >= active_row_ratio)),
            float(np.mean(change_mask.mean(axis=0) >= active_column_ratio)),
        )

    changed_ratio, row_coverage, column_coverage = coverage(mask)
    residual_changed_ratio, residual_row_coverage, residual_column_coverage = coverage(
        residual_mask
    )
    if np.any(highlight_rows):
        highlight_column_coverage = float(
            np.mean(
                highlight_mask[highlight_rows].mean(axis=0)
                >= active_column_ratio
            )
        )
    else:
        highlight_column_coverage = 0.0
    return TableChangeGeometry(
        changed_ratio=changed_ratio,
        row_coverage=row_coverage,
        column_coverage=column_coverage,
        highlight_changed_ratio=float(highlight_mask.mean()),
        highlight_row_coverage=float(highlight_rows.mean()),
        highlight_column_coverage=highlight_column_coverage,
        residual_changed_ratio=residual_changed_ratio,
        residual_row_coverage=residual_row_coverage,
        residual_column_coverage=residual_column_coverage,
    )


def presentation_change_geometry(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_ratio: float = 0.04,
    top_ratio: float = 0.02,
    right_ratio: float = 0.98,
    bottom_ratio: float = 0.90,
    pixel_threshold: int = 12,
    grid: int = 8,
    active_block_ratio: float = 0.02,
) -> PresentationChangeGeometry:
    """Describe content change while excluding common player and slide chrome.

    A presentation build is usually spatially sparse and retains the older
    slide's edges.  A page change is spread across many content blocks and
    destroys a material portion of those edges.  One-pixel edge tolerance
    absorbs antialiasing and tiny rendering shifts without hiding genuine
    additions or removals.
    """
    if left.shape != right.shape:
        raise ValueError("Presentation-change inputs must have the same shape")
    if left.ndim != 2:
        raise ValueError("Presentation-change inputs must be grayscale images")
    if not 0 <= left_ratio < right_ratio <= 1:
        raise ValueError("Presentation horizontal ROI must satisfy 0 <= left < right <= 1")
    if not 0 <= top_ratio < bottom_ratio <= 1:
        raise ValueError("Presentation vertical ROI must satisfy 0 <= top < bottom <= 1")
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("Pixel threshold must be between 0 and 255")
    if grid <= 0:
        raise ValueError("Presentation block grid must be positive")
    if not 0 <= active_block_ratio <= 1:
        raise ValueError("Active block ratio must be between 0 and 1")

    height, width = left.shape
    x0 = min(width - 1, int(round(width * left_ratio)))
    x1 = max(x0 + 1, min(width, int(round(width * right_ratio))))
    y0 = min(height - 1, int(round(height * top_ratio)))
    y1 = max(y0 + 1, min(height, int(round(height * bottom_ratio))))
    left_roi = left[y0:y1, x0:x1]
    right_roi = right[y0:y1, x0:x1]

    absolute_difference = cv2.absdiff(left_roi, right_roi)
    change_mask = absolute_difference >= pixel_threshold
    soft_threshold = max(1, pixel_threshold // 2)
    soft_change_mask = absolute_difference >= soft_threshold
    active_blocks = 0
    block_count = 0
    roi_height, roi_width = change_mask.shape
    for row in range(grid):
        block_y0 = row * roi_height // grid
        block_y1 = (row + 1) * roi_height // grid
        for column in range(grid):
            block_x0 = column * roi_width // grid
            block_x1 = (column + 1) * roi_width // grid
            block = change_mask[block_y0:block_y1, block_x0:block_x1]
            if block.size == 0:
                continue
            block_count += 1
            if float(block.mean()) >= active_block_ratio:
                active_blocks += 1

    left_edges = cv2.Canny(left_roi, 80, 160) > 0
    right_edges = cv2.Canny(right_roi, 80, 160) > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    left_edge_neighbourhood = cv2.dilate(left_edges.astype(np.uint8), kernel) > 0
    right_edge_neighbourhood = cv2.dilate(right_edges.astype(np.uint8), kernel) > 0
    lost_edges = np.logical_and(left_edges, np.logical_not(right_edge_neighbourhood))
    added_edges = np.logical_and(right_edges, np.logical_not(left_edge_neighbourhood))
    changed_edges = np.logical_or(lost_edges, added_edges)
    old_edge_count = int(left_edges.sum())
    retained_edge_ratio = (
        1.0 - float(lost_edges.sum()) / old_edge_count
        if old_edge_count
        else 1.0
    )

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        change_mask.astype(np.uint8),
        connectivity=8,
    )
    foreground_stats = component_stats[1:]
    if len(foreground_stats):
        largest_position = int(np.argmax(foreground_stats[:, cv2.CC_STAT_AREA]))
        largest = foreground_stats[largest_position]
        largest_component_ratio = float(largest[cv2.CC_STAT_AREA]) / change_mask.size
        largest_component_width_ratio = float(largest[cv2.CC_STAT_WIDTH]) / roi_width
        largest_component_height_ratio = float(largest[cv2.CC_STAT_HEIGHT]) / roi_height
    else:
        largest_component_ratio = 0.0
        largest_component_width_ratio = 0.0
        largest_component_height_ratio = 0.0

    return PresentationChangeGeometry(
        changed_ratio=float(change_mask.mean()),
        soft_changed_ratio=float(soft_change_mask.mean()),
        edge_changed_ratio=float(changed_edges.mean()),
        block_spread=(float(active_blocks) / block_count if block_count else 0.0),
        retained_edge_ratio=retained_edge_ratio,
        edge_loss_ratio=float(lost_edges.mean()),
        edge_gain_ratio=float(added_edges.mean()),
        left_edge_ratio=float(left_edges.mean()),
        right_edge_ratio=float(right_edges.mean()),
        largest_component_ratio=largest_component_ratio,
        largest_component_width_ratio=largest_component_width_ratio,
        largest_component_height_ratio=largest_component_height_ratio,
        component_count=max(0, component_count - 1),
    )


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def comparison(left: np.ndarray, right: np.ndarray, left_hash: int, right_hash: int) -> dict[str, float | int]:
    pdelta = pixel_delta(left, right)
    edelta = edge_delta(left, right)
    hdelta = phash_distance(left_hash, right_hash)
    ssim = global_ssim(left, right)
    local = block_change(left, right)
    score = 0.40 * pdelta + 0.20 * edelta + 0.20 * (hdelta / 64.0) + 0.20 * max(0.0, 1.0 - ssim)
    return {
        "pixel_delta": pdelta,
        "edge_delta": edelta,
        "phash_delta": hdelta,
        "ssim": ssim,
        "block_change": local,
        "score": float(score),
    }
