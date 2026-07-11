#!/usr/bin/env python3
"""
Presentation-friendly iTHOR multi-agent demo.

This is a small "CoELA-style" prototype, not the original CoELA algorithm:
one high-level goal is decomposed into two sub-tasks and assigned to two
iTHOR agents. The video overlays the task assignment, agent roles, and
top-down agent positions so the demo is legible in a 4-minute presentation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai2thor.controller import Controller


Color = Tuple[int, int, int]

RED: Color = (235, 50, 55)
BLUE: Color = (60, 130, 255)
YELLOW: Color = (255, 230, 80)
WHITE: Color = (245, 245, 245)
BLACK: Color = (0, 0, 0)
DARK: Color = (20, 24, 32)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_18 = load_font(18)
FONT_22 = load_font(22)
FONT_28 = load_font(28)
FONT_34 = load_font(34)


def as_events(event: Any) -> List[Any]:
    return list(getattr(event, "events", []) or [event])


def get_agent_event(event: Any, agent_id: int) -> Any:
    events = as_events(event)
    if agent_id < len(events):
        return events[agent_id]
    return events[0]


def get_top_frame(event: Any) -> Optional[np.ndarray]:
    candidates = [event] + as_events(event)
    for ev in candidates:
        frames = getattr(ev, "third_party_camera_frames", None)
        if frames is not None and len(frames) > 0:
            return frames[0]
    return None


def get_position(event: Any, agent_id: int) -> Dict[str, float]:
    ev = get_agent_event(event, agent_id)
    return dict(ev.metadata["agent"]["position"])


def get_rotation(event: Any, agent_id: int) -> Dict[str, float]:
    ev = get_agent_event(event, agent_id)
    return dict(ev.metadata["agent"]["rotation"])


def choose_waypoints(reachable: List[Dict[str, float]], side: str, count: int = 5) -> List[Dict[str, float]]:
    """Pick reachable positions that make left/right assignment visually distinct."""
    xs = [p["x"] for p in reachable]
    min_x, max_x = min(xs), max(xs)
    mid_x = (min_x + max_x) / 2.0

    if side == "left":
        pool = [p for p in reachable if p["x"] <= mid_x]
        pool = sorted(pool, key=lambda p: (p["z"], p["x"]))
    else:
        pool = [p for p in reachable if p["x"] >= mid_x]
        pool = sorted(pool, key=lambda p: (p["z"], -p["x"]))

    if len(pool) < count:
        pool = sorted(reachable, key=lambda p: p["x"], reverse=(side == "right"))

    if len(pool) <= count:
        return [dict(p) for p in pool]

    idxs = np.linspace(0, len(pool) - 1, count).round().astype(int).tolist()
    return [dict(pool[i]) for i in idxs]


def yaw_toward(src: Dict[str, float], dst: Dict[str, float]) -> float:
    dx = dst["x"] - src["x"]
    dz = dst["z"] - src["z"]
    if abs(dx) + abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dz))


def hold_reachable_waypoints(points: List[Dict[str, float]], hold_frames: int) -> List[Dict[str, float]]:
    """
    Repeat actual iTHOR reachable positions instead of interpolating between them.

    Interpolated positions can cut through furniture. That looks bad in a
    presentation because physics objects such as chairs can get pushed around.
    """
    out: List[Dict[str, float]] = []
    for point in points:
        out.extend([dict(point) for _ in range(max(1, hold_frames))])
    return out


def world_to_top_pixel(
    pos: Dict[str, float],
    map_camera: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Approximate top-down orthographic projection for AI2-THOR map camera."""
    width, height = image_size
    cam_pos = map_camera.get("position", {})
    cx = float(cam_pos.get("x", 0.0))
    cz = float(cam_pos.get("z", 0.0))
    ortho = float(map_camera.get("orthographicSize", 5.0))

    half_h = ortho
    half_w = ortho * (width / float(height))

    x = (float(pos["x"]) - cx + half_w) / (2 * half_w)
    y = (cz - float(pos["z"]) + half_h) / (2 * half_h)

    px = int(max(0, min(width - 1, x * width)))
    py = int(max(0, min(height - 1, y * height)))
    return px, py


def draw_marker(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Color, label: str) -> None:
    x, y = xy
    r = 15
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=4)
    draw.text((x + r + 6, y - r - 3), label, fill=WHITE, font=FONT_22, stroke_width=3, stroke_fill=BLACK)


def draw_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], fill: Color, outline: Color = WHITE) -> None:
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=2)


def save_composite(
    event: Any,
    frame_path: Path,
    frame_idx: int,
    total_frames: int,
    map_camera: Dict[str, Any],
    caption: str,
) -> None:
    canvas_w, canvas_h = 1280, 720
    top_h = 500
    bottom_h = canvas_h - top_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), BLACK)
    draw = ImageDraw.Draw(canvas)

    top_arr = get_top_frame(event)
    if top_arr is None:
        top_arr = get_agent_event(event, 0).frame
    top_img = Image.fromarray(top_arr).convert("RGB").resize((canvas_w, top_h), Image.Resampling.LANCZOS)
    canvas.paste(top_img, (0, 0))

    # Dark header and explanation panel.
    draw.rectangle((0, 0, canvas_w, 76), fill=(0, 0, 0, 165))
    draw.text((20, 12), "CoELA-style iTHOR demo: decompose -> assign -> execute", fill=YELLOW, font=FONT_28)
    draw.text((20, 45), caption, fill=WHITE, font=FONT_18)

    draw_box(draw, (830, 90, 1260, 232), DARK)
    draw.text((850, 106), "High-level goal", fill=YELLOW, font=FONT_22)
    draw.text((850, 136), "Find useful objects in the kitchen.", fill=WHITE, font=FONT_18)
    draw.text((850, 166), "Agent 0: left-area exploration", fill=RED, font=FONT_18)
    draw.text((850, 194), "Agent 1: right-area exploration", fill=BLUE, font=FONT_18)

    # Draw agent markers on top view.
    for agent_id, color, name in [(0, RED, "A0"), (1, BLUE, "A1")]:
        pos = get_position(event, agent_id)
        px, py = world_to_top_pixel(pos, map_camera, (canvas_w, top_h))
        draw_marker(draw, (px, py), color, name)

    # Progress bar.
    margin = 20
    bar_w = canvas_w - margin * 2
    bar_y = top_h - 28
    progress = frame_idx / max(1, total_frames - 1)
    draw.rounded_rectangle((margin, bar_y, margin + bar_w, bar_y + 12), radius=6, fill=(40, 40, 40))
    draw.rounded_rectangle((margin, bar_y, margin + int(bar_w * progress), bar_y + 12), radius=6, fill=YELLOW)

    # Bottom agent views.
    for agent_id, x0, color, label in [
        (0, 0, RED, "Agent 0 view - left-area task"),
        (1, canvas_w // 2, BLUE, "Agent 1 view - right-area task"),
    ]:
        ev = get_agent_event(event, agent_id)
        view = Image.fromarray(ev.frame).convert("RGB").resize((canvas_w // 2, bottom_h), Image.Resampling.LANCZOS)
        canvas.paste(view, (x0, top_h))
        draw.rectangle((x0, top_h, x0 + canvas_w // 2, top_h + 42), fill=(0, 0, 0))
        draw.text((x0 + 18, top_h + 10), label, fill=color, font=FONT_22)

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=92)


def create_video(frame_dir: Path, video_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        return False

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        return False

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--output", default="ithor_output_v3")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output).expanduser().resolve()
    frame_dir = out_dir / "frames"
    video_path = out_dir / "ithor_coela_style_demo_v3.mp4"
    plan_path = out_dir / "plan_log.json"

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    controller = Controller(
        scene=args.scene,
        agentCount=2,
        width=args.width,
        height=args.height,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    try:
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]
        if len(reachable) < 8:
            raise RuntimeError(f"Too few reachable positions in {args.scene}: {len(reachable)}")

        event = controller.step(action="GetMapViewCameraProperties")
        map_camera = dict(event.metadata["actionReturn"])
        map_camera["orthographicSize"] = float(map_camera.get("orthographicSize", 5.0)) * 1.08
        event = controller.step(action="AddThirdPartyCamera", **map_camera)

        left_waypoints = choose_waypoints(reachable, "left", count=5)
        right_waypoints = choose_waypoints(reachable, "right", count=5)
        path0 = hold_reachable_waypoints(left_waypoints, args.hold_frames)
        path1 = hold_reachable_waypoints(right_waypoints, args.hold_frames)
        total = min(len(path0), len(path1))

        plan = {
            "demo_type": "CoELA-style task decomposition and assignment prototype on iTHOR",
            "important_note": "This is not the original CoELA algorithm. It is a small rule-based prototype for the middle presentation.",
            "scene": args.scene,
            "high_level_goal": "Find useful objects in the kitchen.",
            "decomposition": [
                {"agent": 0, "subtask": "Explore left area and observe candidate objects."},
                {"agent": 1, "subtask": "Explore right area and observe candidate objects."},
            ],
            "agent_0_waypoints": left_waypoints,
            "agent_1_waypoints": right_waypoints,
        }
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        for i in range(total):
            p0 = path0[i]
            p1 = path1[i]
            n0 = path0[min(i + 1, total - 1)]
            n1 = path1[min(i + 1, total - 1)]

            event = controller.step(
                action="TeleportFull",
                agentId=0,
                x=p0["x"],
                y=p0["y"],
                z=p0["z"],
                rotation={"x": 0, "y": yaw_toward(p0, n0), "z": 0},
                horizon=20,
                standing=True,
                forceAction=False,
            )
            event = controller.step(
                action="TeleportFull",
                agentId=1,
                x=p1["x"],
                y=p1["y"],
                z=p1["z"],
                rotation={"x": 0, "y": yaw_toward(p1, n1), "z": 0},
                horizon=20,
                standing=True,
                forceAction=False,
            )

            if i < total * 0.25:
                caption = "Step 1/3: high-level goal is decomposed into two area-search tasks."
            elif i < total * 0.75:
                caption = "Step 2/3: Agent 0 and Agent 1 execute their assigned waypoints in parallel."
            else:
                caption = "Step 3/3: execution trace is saved for analysis and presentation."

            save_composite(event, frame_dir / f"frame_{i:04d}.jpg", i, total, map_camera, caption)

        video_ok = create_video(frame_dir, video_path, args.fps)
        print(f"[ok] frames: {frame_dir}")
        print(f"[ok] plan log: {plan_path}")
        if video_ok:
            print(f"[ok] video: {video_path}")
        else:
            print("[warn] imageio/imageio-ffmpeg not available; frames were saved.")
            print("       Install with: python -m pip install imageio imageio-ffmpeg")
            print(f"       Then rerun, or use ffmpeg on: {frame_dir}/frame_%04d.jpg")

    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
