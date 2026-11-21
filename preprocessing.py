"""
Preprocessing utilities for the AI Stricture pipeline.

Import into notebooks with:
    from preprocessing import (
        crop_frame, octagon_mask, mean_normalized_red, compute_mnr_series,
        select_segments, run_out_of_body_detector_pipeline,
        laplacian_on_kept, blur_gate, frame_step, main
    )

Frame-selection pipeline (all decisions + reasoning in Decision_Log.md, Decisions 3-11):
    1. compute_mnr_series       per-frame mean_normalized_red on cropped frames, at 25 fps
                            timing: every 2nd frame of 50 fps videos, original frame
                            numbers kept (Decision 14)    
    2. run_pelt                 PELT segmentation (pen=500, min_size=30, jump=1)
    3. filter_by_stability      keep flat segments (90-10 percentile range <= 10)
    4. remove_out_of_body_segments   drop segments with median < 100
    5. filter_by_containing_baseline keep segments whose 10-90 range contains the
                                     baseline (tolerance=1)
    6. rescue_with_tolerance    passive backstop if kept yield <= 15% of video frames
                                                                
    7. blur_gate                Blur gate (Decision 13), runs on the out-of-body result:
                                Laplacian variance at 1/2 resolution inside the octagon
                                mask; drop the blurriest 20% of kept frames per video

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

MASK_TOP_X = (213, 1136)      # Decision 12: octagon mask, start/end x on the top and bottom row
MASK_SIDE_Y = (265, 815)      # Decision 12: octagon mask, rows where the corner cuts end on the sides

BLUR_DOWNSCALE = 2            # Decision 13: blur score computed at 1/2 resolution
BLUR_DROP_PCT = 20            # Decision 13: drop the blurriest 20% of kept frames per video

TARGET_FPS = 25               # Decision 14: every video is analysed at 25 fps timing (50 fps -> every 2nd frame)
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
    return frame[:, x_start:x_end + 1]

def octagon_mask(height=1080, width=1350, top_x=MASK_TOP_X, side_y=MASK_SIDE_Y):
    """
    Boolean mask of the usable image area inside the octagon, for a cropped frame
    (crop_frame output, 1350 x 1080). True = pixel used for the blur score.

    Polygon through the mask points in Decision_Log.md (Decision 12): the measured
    octagon (202/1147, 252/828) pulled in 8 px, so every True pixel is at least
    8 px from the octagon's black border (where the border no longer affects the
    Laplacian). Mask pixels: 1,344,897.
    """
    x0, x1 = top_x
    y0, y1 = side_y
    last_x, last_y = width - 1, height - 1
    points = np.array([(x0, 0), (x1, 0),                 # top edge
                       (last_x, y0), (last_x, y1),       # right side
                       (x1, last_y), (x0, last_y),       # bottom edge
                       (0, y1), (0, y0)], np.int32)      # left side
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)
    
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


def frame_step(fps, target_fps=TARGET_FPS):
    """
    How many original frames make one analysed frame (Decision 14):
    1 for a 25 fps video (every frame), 2 for a 50 fps video (every 2nd frame).
    """
    if not fps or np.isnan(fps) or fps <= 0:
        print(f"  fps could not be read ({fps}) -- using every frame")
        return 1

    step = max(1, int(round(fps / target_fps)))
    if abs(fps / step - target_fps) > 1:
        print(f"  warning: {fps:.2f} fps is not a multiple of {target_fps} fps -- "
              f"analysing at {fps / step:.2f} fps")
    return step


def compute_mnr_series(video_path, target_fps=TARGET_FPS):
    """
    Apply crop_frame + mean_normalized_red to the video at 25 fps timing (Decision 14):
    every frame of a 25 fps video, every 2nd frame (0, 2, 4, ...) of a 50 fps video.
    The skipped frames are only grabbed (moved past), not decoded into an image.

    Returns (values, frame_numbers, fps, step):
        values          one mean_normalized_red value per analysed frame (NaN where a frame failed)
        frame_numbers   original frame number of each value (same order) -- never renumbered
        fps             the video's own frame rate (cv2.CAP_PROP_FPS)
        step            1 for 25 fps, 2 for 50 fps
    or None if the video can't be opened / has no readable frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  Could not open — skipping")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = frame_step(fps, target_fps)

    values, frame_numbers = [], []
    idx = 0                                   # original frame number

    while True:
        if not cap.grab():                    # move to the next frame
            break
        if idx % step == 0:                   # analysed frame: decode it
            ret, frame = cap.retrieve()
            if not ret:
                break
            try:
                cropped = crop_frame(frame)
                values.append(mean_normalized_red(cropped))
            except ValueError as e:
                print(f"  frame {idx}: crop failed ({e}) — skipping")
                values.append(np.nan)
            frame_numbers.append(idx)
        idx += 1

    cap.release()

    if not values or np.all(np.isnan(values)):
        print("  No usable frames read — skipping")
        return None

    print(f"fps={fps:.0f}, step={step}: analysed {len(values)} of {idx} frames  |  "
          f"min={np.nanmin(values):.1f}  max={np.nanmax(values):.1f}  mean={np.nanmean(values):.1f}")

    return np.array(values), np.array(frame_numbers), fps, step


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

def plot_mean_normalized_red(video_path, values, frame_numbers=None):
    x = np.arange(len(values)) if frame_numbers is None else frame_numbers
    plt.figure(figsize=(14, 5))
    plt.plot(x, values)
    plt.title(f"{video_path.split(chr(92))[-1]} — mean_normalized_red per analysed frame")
    plt.xlabel("original frame number")
    plt.ylabel("mean_normalized_red (0-255)")
    plt.grid(alpha=0.3)
    plt.show()


def plot_pelt(video_path, values, bkps, pen, frame_numbers=None):
    """Plot the signal with vertical lines at each PELT breakpoint (x = original frame number;
    each line sits on the first frame of the next segment)."""
    x = np.arange(len(values)) if frame_numbers is None else frame_numbers
    bkp_frames = [int(x[b]) for b in bkps[:-1]]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, values, linewidth=0.8)
    for f in bkp_frames:
        ax.axvline(x=f, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{video_path.split(chr(92))[-1]}  pen={pen}  breakpoints (frames): {bkp_frames}")
    ax.set_xlabel("original frame number")
    ax.set_ylabel("mean_normalized_red")
    plt.tight_layout()
    plt.show()


def shading_plot(values, all_segs, shaded, title="Selected segments shaded green", frame_numbers=None):
    """Plot the signal, gray dotted lines at every segment start, green shading on `shaded`.
    x = original frame number; segments are drawn from their first_frame / last_frame."""
    x = np.arange(len(values)) if frame_numbers is None else frame_numbers
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, values, linewidth=0.8, color="tab:blue")
    
    for s in all_segs:
        ax.axvline(x=s["first_frame"], color="gray", linestyle=":", linewidth=0.8)
    
    for s in shaded:
        ax.axvspan(s["first_frame"], s["last_frame"], color="green", alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel("original frame number")
    ax.set_ylabel("mean_normalized_red")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core pipeline (no video I/O, no plotting)
# ---------------------------------------------------------------------------

def select_segments(values, pen=PELT_PEN, frame_numbers=None):
    """
    Steps 2-6 of the frame-selection pipeline on an already-computed
    mean_normalized_red series. Pure function: no video I/O, no plotting, no prints.

    frame_numbers: original frame number of each value (from compute_mnr_series).
    If None, 0, 1, 2, ... is used (every frame analysed).

    Every segment dict has two kinds of fields:
        start, end       POSITIONS in `values` (end exclusive) -- internal, used only
                         to slice `values` inside this pipeline
        frames           ORIGINAL frame numbers of the segment's frames
        first_frame      ORIGINAL frame number of its first frame (inclusive)
        last_frame       ORIGINAL frame number of its last frame (inclusive)
    Anything that fetches frames from the video, prints or plots must use
    frames / first_frame / last_frame, never start / end (Decision 14).

    Returns a dict with: bkps (positions), all_segs, survivors, clean_survivors, dropped_oob,
    kept (final segments), baseline (None if nothing survived), pct_before_rescue,
    pct_kept (final), rescue_triggered, final_tolerance, frame_numbers.

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

    frame_numbers = np.arange(len(values)) if frame_numbers is None else np.asarray(frame_numbers)
    if len(frame_numbers) != len(values):
        raise ValueError(f"frame_numbers has {len(frame_numbers)} entries but values has {len(values)}.")

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

    # 8. original frame numbers for every segment (Decision 14). survivors, clean_survivors,
    #    dropped_oob and kept hold the same segment dicts as all_segs, so they all get them.
    for s in all_segs:
        s["frames"] = frame_numbers[s["start"]:s["end"]].tolist()
        s["first_frame"] = s["frames"][0]
        s["last_frame"] = s["frames"][-1]

    return {
        "bkps": bkps, "all_segs": all_segs, "survivors": survivors,
        "clean_survivors": clean_survivors, "dropped_oob": dropped_oob,
        "kept": kept, "baseline": baseline,
        "pct_before_rescue": pct_before_rescue, "pct_kept": pct,
        "rescue_triggered": rescue_triggered, "final_tolerance": final_tolerance,
        "frame_numbers": frame_numbers,
    }


# ---------------------------------------------------------------------------
# End-to-end with prints + plots (for notebook investigation)
# ---------------------------------------------------------------------------

def run_out_of_body_detector_pipeline(video_path, pen=PELT_PEN, plot=False):
    # 1. per-frame mean_normalized_red at 25 fps timing (Decision 14)
    series = compute_mnr_series(video_path)
    if series is None:
        return None
    values, frame_numbers, fps, step = series
    if plot:
        plot_mean_normalized_red(video_path, values, frame_numbers)

    # 2-6. segmentation, filters, baseline, rescue
    #      every segment gets frames / first_frame / last_frame = original frame numbers
    r = select_segments(values, pen=pen, frame_numbers=frame_numbers)
    if plot:
        plot_pelt(video_path, values, r["bkps"], pen, frame_numbers)

    print("All segments (original frame numbers, first-last inclusive):")
    
    for s in r["all_segs"]:
        print(f"  {s['first_frame']:>4}-{s['last_frame']:<4} {s['length']*step/fps:5.2f}s mean={s['mean']:6.1f} range_90_10={s['range_90_10']:5.1f}")
    print(f"\nSurvivors (range_90_10<={MAX_RANGE_90_10}):")
    
    for s in r["survivors"]:
        print(f"  {s['first_frame']:>4}-{s['last_frame']:<4} {s['length']*step/fps:5.2f}s mean={s['mean']:6.1f} range_90_10={s['range_90_10']:5.1f}")
    
    if plot:
        shading_plot(values, r["all_segs"], r["survivors"],
                     title=f"Flatness survivors shaded green (pen={pen}, range_90_10<={MAX_RANGE_90_10})",
                     frame_numbers=frame_numbers)

    if not r["dropped_oob"]:
        print(f"\nDropped as out-of-body (segment median < {OUT_OF_BODY_THRESHOLD}): None")
    else:
        print(f"\nDropped as out-of-body (segment median < {OUT_OF_BODY_THRESHOLD}):")
        for s in r["dropped_oob"]:
            print(f"  {s['first_frame']:>4}-{s['last_frame']:<4} median={s['segment_median']:6.1f}")

    if r["baseline"] is None:
        print("\nNo segments survived flatness + out-of-body filters -- nothing kept.")
    else:
        print("\nbaseline (median of pooled non-OOB survivor frames, top-10% trimmed):", round(r["baseline"], 1))
        for s in r["clean_survivors"]:
            print(f"  {s['first_frame']:>4}-{s['last_frame']:<4}  range=[{s['p_low']:.1f}, {s['p_high']:.1f}]  contains_baseline={s['contains_baseline']}")

    total_frames = len(values)

    print(f"\nKept frames (before rescue): {r['pct_before_rescue']:.1f}% of {total_frames} analysed frames")
    if r["rescue_triggered"]:
        print(f"Yield was <={RESCUE_TARGET_PCT}% -- rescue triggered, widened tolerance to {r['final_tolerance']}")
    else:
        print(f"Yield above {RESCUE_TARGET_PCT}% -- rescue not needed")

    kept_frames = sum(len(s["frames"]) for s in r["kept"])
    
    print("Kept (final, original frame numbers, first-last inclusive):",
          [(s["first_frame"], s["last_frame"]) for s in r["kept"]])
    print(f"Kept frames (final): {kept_frames} / {total_frames} analysed frames  ({r['pct_kept']:.1f}%)")

    if plot:
        shading_plot(values, r["all_segs"], r["kept"],
                     title=f"Final kept segments shaded green (pen={pen})", frame_numbers=frame_numbers)

    r["values"] = values
    r["fps"] = fps
    r["step"] = step
    r["video_path"] = video_path
    
    return r


# ---------------------------------------------------------------------------
# Blur gate (Decision 13)
# ---------------------------------------------------------------------------

def laplacian_on_kept(oob_result, mask=None, downscale=BLUR_DOWNSCALE):
    """
    Blur score for the frames kept by the out-of-body detector only.

    Each cropped grayscale frame is shrunk by `downscale` (INTER_AREA averages each
    2x2 block, removing pixel-level noise and compression artefacts), then
    score = variance of the 3x3 Laplacian inside the octagon mask (Decision 12),
    shrunk to the same size and eroded by 1 px. No normalization. Low = blurry.

    Returns (kept_idx, scores): frame numbers and their scores, same order.
    """
        
    if mask is None:
        mask = octagon_mask()
    h, w = mask.shape
    size = (w // downscale, h // downscale)                       # (675, 540) for downscale=2
    small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.erode(small_mask, np.ones((3, 3), np.uint8)).astype(bool)

    # original frame numbers of every kept frame (Decision 14)
    kept_idx = np.array(sorted(f for s in oob_result["kept"] for f in s["frames"]))    
    wanted = set(kept_idx.tolist())
    
    scores = {}
    cap = cv2.VideoCapture(oob_result["video_path"])
    i = 0

    while len(scores) < len(wanted):
        if i in wanted:
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(crop_frame(frame), cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
            lap = cv2.Laplacian(g, cv2.CV_64F)
            scores[i] = float(lap[small_mask].var())
        else:
            if not cap.grab():                                    # skip frame without converting it
                break
        i += 1
    cap.release()

    return kept_idx, np.array([scores.get(i, np.nan) for i in kept_idx])


def blur_gate(oob_result, mask=None, drop_pct=BLUR_DROP_PCT):
    """
    Blur gate: score the out-of-body survivors and drop the blurriest `drop_pct`%
    of them, per video (Decision 13).

    Returns a dict:
        video_path     same video, so the next stage only needs this result
        frames         frame numbers that pass the gate (sorted)
        scores         {frame number: blur score} for every scored frame -- kept so the
                       final picker can rank by sharpness
        threshold      score at the drop_pct percentile; frames below it are dropped
        n_in, n_out    number of frames scored / passing
    """
    kept_idx, scores = laplacian_on_kept(oob_result, mask=mask)
    valid = ~np.isnan(scores)
    threshold = float(np.percentile(scores[valid], drop_pct))
    passing = kept_idx[valid & (scores >= threshold)]

    return {
        "video_path": oob_result["video_path"],
        "frames": passing.tolist(),
        "scores": {int(f): float(s) for f, s in zip(kept_idx, scores) if not np.isnan(s)},
        "threshold": threshold,
        "n_in": int(valid.sum()),
        "n_out": int(len(passing)),
    }



# ---------------------------------------------------------------------------
# Main: full preprocessing for one video
# ---------------------------------------------------------------------------

def main(video_path, plot=False):
    """
    Run the whole preprocessing pipeline on one video:

        1. run_out_of_body_detector_pipeline   out-of-body detection (Decisions 3-11)
        2. octagon_mask                        creates a mask for octagon black region
        3. blur_gate                           drop the blurriest 20% of kept frames (Decisions 12-13)

    plot=True shows the out-of-body detector's plots.
    Returns a dict: video_path, out_of_body (detector result), blur (blur_gate result),
    frames (frame numbers that passed every stage, sorted) -- or None if the video
    could not be read.
    """
    # 1. out-of-body detection
    oob = run_out_of_body_detector_pipeline(video_path, plot=plot)
    if oob is None:
        print(f"Could not read video: {video_path}")
        return None

    # 2. creates a mask for black region on all four corners of frame
    mask = octagon_mask()

    # 3. blur gate on the kept frames
    blur = blur_gate(oob, mask)
    print(f"\nBlur gate: {blur['n_in']} scored, {blur['n_out']} pass "
          f"(dropped {blur['n_in'] - blur['n_out']} below score {blur['threshold']:.1f})")

    return {
        "video_path": video_path,
        "out_of_body": oob,
        "blur": blur,
    }