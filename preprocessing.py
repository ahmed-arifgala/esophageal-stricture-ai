"""
Preprocessing utilities for the AI Stricture pipeline.

Import into notebooks with:
    from preprocessing import (
        crop_frame, mean_normalized_red, compute_mnr_series,
        select_segments, run_out_of_body_detector_pipeline,
    )

Frame-selection pipeline (all decisions + reasoning in Decision_Log.md, Decisions 3-11):
    1. compute_mnr_series       per-frame mean_normalized_red on cropped frames
    2. run_pelt                 PELT segmentation (pen=500, min_size=30, jump=1)
    3. filter_by_stability      keep flat segments (90-10 percentile range <= 10)
    4. remove_out_of_body_segments   drop segments with median < 100
    5. filter_by_containing_baseline keep segments whose 10-90 range contains the
                                     baseline (tolerance=1)
    6. rescue_with_tolerance    passive backstop if kept yield <= 15% of video frames

`select_segments(values)` runs steps 2-6 on an already-computed series (no video I/O,
no plotting). `run_out_of_body_detector_pipeline(video_path)` runs everything end to end with prints/plots.
"""

import numpy as np
import cv2
import ruptures as rpt
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Finalized parameters (see Decision_Log.md)
# ---------------------------------------------------------------------------
PELT_PEN = 500            # Decision 6
PELT_MIN_SIZE = 30        # Decision 5
PELT_JUMP = 1             # Decision 5
MAX_RANGE_90_10 = 10      # Decision 7
OUT_OF_BODY_THRESHOLD = 100   # Decision 8
BASELINE_TOP_TRIM_PCT = 90    # Decision 9
CONTAINMENT_TOLERANCE = 1     # Decision 10
RESCUE_TARGET_PCT = 15        # Decision 11


# ---------------------------------------------------------------------------
# Per-frame utilities
# ---------------------------------------------------------------------------

def crop_frame(frame, x_start=550, x_end=1899, expected_width=1920):
    """
    Apply the finalized fixed horizontal crop to a single frame.
    Removes the black surround / patient-data text panel on the left,
    and the black margin on the right.

    Bounds finalized in Decision_Log.md (Decision 2): x_start=550, x_end=1899,
    cross-validated across 8 videos / 5 patients at 1920x1080 resolution
    using two independent methods.
    """
    h, w = frame.shape[:2]
    if w != expected_width:
        raise ValueError(
            f"Frame width is {w}px, expected {expected_width}px. "
            f"Fixed crop bounds (x_start={x_start}, x_end={x_end}) were derived "
            f"specifically for {expected_width}px-wide frames and may not be valid here — "
            f"re-verify before applying."
        )
    return frame[:, x_start:x_end]


def mean_normalized_red(frame, min_brightness=10, epsilon=1e-6):
    """
    Compute mean-normalized-red for a frame: the average, per pixel, of how
    red-dominant that pixel's color is -- r / (r+g+b), scaled to 0-255.

    This is independent of raw brightness, unlike a simple mean-red check:
    a bright white pixel and a dim red pixel are told apart correctly here,
    where a naive brightness-based check would not (see Research_on_Frame_Selection.md,
    Stanek et al. 2012 entry).

    Pixels whose total brightness (r+g+b) is below `min_brightness` are
    excluded before averaging. These are near-pure-black pixels, where
    r/(r+g+b) is not a meaningful color signal.

    NOTE: this does NOT remove the black corner triangles left after
    crop_frame(). Checked on a real frame: corner pixels have r+g+b ~46-48
    (not near 0), so 0% of pixels were excluded and the corners stay in the
    average, each scoring ~85 (r ~= g ~= b). This adds a roughly constant
    downward offset to every frame's score. Deliberately left as-is: all
    pipeline thresholds were calibrated with this offset included (see
    Production_Limitations.md).

    Expects a BGR frame (OpenCV's default channel order).

    Returns a float in [0, 255]. Higher = more red-dominant / tissue-like.
    Returns 0.0 if the frame has no valid (non-near-black) pixels at all.
    """
    frame = frame.astype(np.float32)
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    total = r + g + b
    valid = total >= min_brightness

    if not np.any(valid):
        return 0.0

    ratio = np.zeros_like(total)
    ratio[valid] = r[valid] / (total[valid] + epsilon)

    pixel_score = 255.0 * ratio
    return float(pixel_score[valid].mean())


def compute_mnr_series(video_path):
    """Apply crop_frame + mean_normalized_red to every frame of a video.
    Returns a numpy array of values (one per frame, NaN where a frame failed),
    or None if the video can't be opened / has no readable frames."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  Could not open — skipping")
        return None

    values = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            cropped = crop_frame(frame)
            values.append(mean_normalized_red(cropped))
        except ValueError as e:
            print(f"  frame {idx}: crop failed ({e}) — skipping")
            values.append(np.nan)
        idx += 1

    cap.release()

    if not values or np.all(np.isnan(values)):
        print("  No usable frames read — skipping")
        return None

    print(f"frames analyzed: {len(values)}  |  min={np.nanmin(values):.1f}  "
          f"max={np.nanmax(values):.1f}  mean={np.nanmean(values):.1f}")

    return np.array(values)


# ---------------------------------------------------------------------------
# Segmentation + segment filters
# ---------------------------------------------------------------------------

def run_pelt(values, pen=PELT_PEN):
    """Run PELT segmentation on the mean_normalized_red series, return raw breakpoints
    (last value is always len(values), not a real breakpoint)."""

    signal = np.array(values)
    algo = rpt.Pelt(model="l2", min_size=PELT_MIN_SIZE, jump=PELT_JUMP).fit(signal)
    return algo.predict(pen=pen)


def filter_by_stability(values, bkps, max_range=MAX_RANGE_90_10, low_pct=10, high_pct=90):
    """
    Drop segments that are not genuinely flat (90-10 percentile range too wide).
    No length filter -- deliberately (Decision 7).
    Returns (survivors, all_segments).
    """
    values = np.asarray(values)
    bkps = list(bkps)

    if bkps[-1] != len(values):
        bkps.append(len(values))
    
    starts = [0] + bkps[:-1]
    ends = bkps

    all_segments = []
    
    for start, end in zip(starts, ends):
        seg = values[start:end]
        p_low, p_high = np.percentile(seg, [low_pct, high_pct])
        all_segments.append({
            "start": start, "end": end, "length": end - start,
            "mean": float(seg.mean()),
            "range_90_10": float(p_high - p_low),
        })
    survivors = [s for s in all_segments if s["range_90_10"] <= max_range]
    return survivors, all_segments


def remove_out_of_body_segments(values, survivors, out_of_body_threshold=OUT_OF_BODY_THRESHOLD):
    """
    Drop any surviving segment whose own median falls below out_of_body_threshold --
    equivalent to "more than 50% of this segment's frames are out-of-body" (median
    below the cutoff means the majority of the segment sits below it). Keeps an
    out-of-body-but-flat segment from contaminating the baseline calculation next.
    Returns (clean, dropped).
    """
    values = np.asarray(values)
    clean, dropped = [], []

    for s in survivors:
        seg = values[s["start"]:s["end"]]
        seg_median = float(np.median(seg))
        s["segment_median"] = seg_median
        (dropped if seg_median < out_of_body_threshold else clean).append(s)

    return clean, dropped


def filter_by_containing_baseline(values, clean_survivors, low_pct=10, high_pct=90,
                                  top_trim_pct=BASELINE_TOP_TRIM_PCT,
                                  tolerance=CONTAINMENT_TOLERANCE):
    """
    Baseline = median of pooled per-frame values from just the surviving,
    non-out-of-body segments, with the top 10% trimmed off first -- protects
    the baseline from a stable-but-too-red segment (glare/close-up) skewing it
    upward. No floor cleaning needed here -- out-of-body is already handled
    by remove_out_of_body_segments before this runs.

    A segment is kept if its own value range (10-90 percentile), widened by
    `tolerance` on each side, contains the baseline. Mutates each segment
    dict in `clean_survivors` in place (adds p_low/p_high/contains_baseline).

    Returns (kept, baseline). If `clean_survivors` is empty, returns ([], None).
    """
    if not clean_survivors:
        return [], None

    values = np.asarray(values)
    pooled = np.concatenate([values[s["start"]:s["end"]] for s in clean_survivors])
    p_top = np.percentile(pooled, top_trim_pct)
    pooled_trimmed = pooled[pooled <= p_top]
    baseline = float(np.median(pooled_trimmed))

    kept = []
    for s in clean_survivors:
        seg = values[s["start"]:s["end"]]
        p_low, p_high = np.percentile(seg, [low_pct, high_pct])
        s["p_low"], s["p_high"] = float(p_low), float(p_high)
        s["contains_baseline"] = bool((p_low - tolerance) <= baseline <= (p_high + tolerance))
        if s["contains_baseline"]:
            kept.append(s)
    return kept, baseline


def rescue_with_tolerance(values, clean_survivors, baseline, kept, target_pct=RESCUE_TARGET_PCT,
                          tolerance_start=CONTAINMENT_TOLERANCE, tolerance_step=0.5, max_tolerance=5,
                          low_pct=10, high_pct=90):
    """
    If kept-frame yield is <= target_pct of total video frames, progressively
    widen the baseline-containment tolerance and rescue the minimum number of
    additional segments needed to clear the threshold -- closest-to-baseline
    segments first, one at a time, rather than pulling in everything that
    newly qualifies at a given tolerance all at once.
    """
    values = np.asarray(values)
    total_frames = len(values)
    kept = list(kept)
    kept_ids = {(s["start"], s["end"]) for s in kept}
    tolerance = tolerance_start
    pending = []

    def pct_kept():
        return 100 * sum(s["end"] - s["start"] for s in kept) / total_frames

    while pct_kept() <= target_pct:
        if not pending:
            tolerance += tolerance_step
            if tolerance > max_tolerance:
                break  # safety cap -- stop trying to rescue further

            newly_qualified = []
            for s in clean_survivors:
                sid = (s["start"], s["end"])
                if sid in kept_ids:
                    continue
                seg = values[s["start"]:s["end"]]
                p_low, p_high = np.percentile(seg, [low_pct, high_pct])
                if (p_low - tolerance) <= baseline <= (p_high + tolerance):
                    newly_qualified.append(s)

            if not newly_qualified:
                continue  # nothing new at this tolerance, keep widening

            newly_qualified.sort(key=lambda s: abs(s["mean"] - baseline))
            pending = newly_qualified

        s = pending.pop(0)
        kept.append(s)
        kept_ids.add((s["start"], s["end"]))

    return kept, tolerance, round(pct_kept(), 1)




# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib imported lazily so the module imports without it)
# ---------------------------------------------------------------------------

def plot_mean_normalized_red(video_path, values):
    plt.figure(figsize=(14, 5))
    plt.plot(values)
    plt.title(f"{video_path.split(chr(92))[-1]} — mean_normalized_red per frame")
    plt.xlabel("frame index")
    plt.ylabel("mean_normalized_red (0-255)")
    plt.grid(alpha=0.3)
    plt.show()


def plot_pelt(video_path, values, bkps, pen):
    """Plot the signal with vertical lines at each PELT breakpoint."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(values, linewidth=0.8)
    for bkp in bkps[:-1]:
        ax.axvline(x=bkp, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{video_path.split(chr(92))[-1]}  pen={pen}  breakpoints: {bkps[:-1]}")
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean_normalized_red")
    plt.tight_layout()
    plt.show()


def shading_plot(values, all_segs, shaded, title="Selected segments shaded green"):
    """Plot the signal, gray dotted lines at every segment start, green shading on `shaded`."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(values, linewidth=0.8, color="tab:blue")
    
    for s in all_segs:
        ax.axvline(x=s["start"], color="gray", linestyle=":", linewidth=0.8)
    
    for s in shaded:
        ax.axvspan(s["start"], s["end"], color="green", alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean_normalized_red")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core pipeline (no video I/O, no plotting)
# ---------------------------------------------------------------------------

def select_segments(values, pen=PELT_PEN):
    """
    Steps 2-6 of the frame-selection pipeline on an already-computed
    mean_normalized_red series. Pure function: no video I/O, no plotting, no prints.

    Returns a dict with: bkps, all_segs, survivors, clean_survivors, dropped_oob,
    kept (final segments, list of dicts with start/end), baseline (None if nothing
    survived), pct_before_rescue, pct_kept (final), rescue_triggered, final_tolerance.

    Raises ValueError if `values` contains NaN (frames whose crop failed) -- PELT and
    the percentile steps can't handle NaN; the caller must decide how to treat them.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Empty values series.")
    if np.isnan(values).any():
        raise ValueError(
            f"{int(np.isnan(values).sum())} NaN value(s) in series (frames whose crop failed). "
            f"Handle them before segmenting."
        )
    if values.size < PELT_MIN_SIZE:
        raise ValueError(
            f"Series has only {values.size} frames; PELT needs at least min_size={PELT_MIN_SIZE}."
        )

    total_frames = len(values)

    # 2. PELT segmentation
    bkps = run_pelt(values, pen=pen) # returns breakpoints

    # 3. flatness filter (also uses breakpoints to create segments)
    survivors, all_segs = filter_by_stability(values, bkps)
    
    # 4. drop out-of-body segments (median < 100)
    clean_survivors, dropped_oob = remove_out_of_body_segments(values, survivors)
    
    # 5. baseline containment filter
    kept, baseline = filter_by_containing_baseline(values, clean_survivors)

    # 6. yield percentage
    pct = 100 * sum(s["end"] - s["start"] for s in kept) / total_frames
    pct_before_rescue = pct
    final_tolerance = CONTAINMENT_TOLERANCE
    rescue_triggered = False

    # 7. rescue -- only fires if yield is too low
    if pct <= RESCUE_TARGET_PCT and clean_survivors:
        rescue_triggered = True
        kept, final_tolerance, pct = rescue_with_tolerance(values, clean_survivors, baseline, kept)

    return {
        "bkps": bkps, "all_segs": all_segs, "survivors": survivors,
        "clean_survivors": clean_survivors, "dropped_oob": dropped_oob,
        "kept": kept, "baseline": baseline,
        "pct_before_rescue": pct_before_rescue, "pct_kept": pct,
        "rescue_triggered": rescue_triggered, "final_tolerance": final_tolerance,
    }


# ---------------------------------------------------------------------------
# End-to-end with prints + plots (for notebook investigation)
# ---------------------------------------------------------------------------

def run_out_of_body_detector_pipeline(video_path, pen=PELT_PEN, plot=False):
    # 1. per-frame mean_normalized_red
    values = compute_mnr_series(video_path)
    if values is None:
        return None
    if plot:
        plot_mean_normalized_red(video_path, values)

    # 2-6. segmentation, filters, baseline, rescue
    r = select_segments(values, pen=pen)
    if plot:
        plot_pelt(video_path, values, r["bkps"], pen)

    print("All segments:")
    
    for s in r["all_segs"]:
        print(f"  {s['start']:>4}-{s['end']:<4} len={s['length']:<4} mean={s['mean']:6.1f} range_90_10={s['range_90_10']:5.1f}")
    print(f"\nSurvivors (range_90_10<={MAX_RANGE_90_10}):")
    
    for s in r["survivors"]:
        print(f"  {s['start']:>4}-{s['end']:<4} len={s['length']:<4} mean={s['mean']:6.1f} range_90_10={s['range_90_10']:5.1f}")
    
    if plot:
        shading_plot(values, r["all_segs"], r["survivors"],
                     title=f"Flatness survivors shaded green (pen={pen}, range_90_10<={MAX_RANGE_90_10})")

    if not r["dropped_oob"]:
        print(f"\nDropped as out-of-body (segment median < {OUT_OF_BODY_THRESHOLD}): None")
    else:
        print(f"\nDropped as out-of-body (segment median < {OUT_OF_BODY_THRESHOLD}):")
        for s in r["dropped_oob"]:
            print(f"  {s['start']:>4}-{s['end']:<4} median={s['segment_median']:6.1f}")

    if r["baseline"] is None:
        print("\nNo segments survived flatness + out-of-body filters -- nothing kept.")
    else:
        print("\nbaseline (median of pooled non-OOB survivor frames, top-10% trimmed):", round(r["baseline"], 1))
        for s in r["clean_survivors"]:
            print(f"  {s['start']:>4}-{s['end']:<4}  range=[{s['p_low']:.1f}, {s['p_high']:.1f}]  contains_baseline={s['contains_baseline']}")

    total_frames = len(values)

    print(f"\nKept frames (before rescue): {r['pct_before_rescue']:.1f}% of {total_frames}")
    if r["rescue_triggered"]:
        print(f"Yield was <={RESCUE_TARGET_PCT}% -- rescue triggered, widened tolerance to {r['final_tolerance']}")
    else:
        print(f"Yield above {RESCUE_TARGET_PCT}% -- rescue not needed")

    kept_frames = sum(s["end"] - s["start"] for s in r["kept"])
    
    print("Kept (final):", [(s["start"], s["end"]) for s in r["kept"]])
    print(f"Kept frames (final): {kept_frames} / {total_frames}  ({r['pct_kept']:.1f}%)")

    if plot:
        shading_plot(values, r["all_segs"], r["kept"],
                     title=f"Final kept segments shaded green (pen={pen})")

    r["values"] = values
    
    return r
