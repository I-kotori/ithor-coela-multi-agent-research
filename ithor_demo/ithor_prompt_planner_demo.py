#!/usr/bin/env python3
"""
Prompt-engineering style multi-agent planner demo for AI2-THOR/iTHOR.

Purpose
-------
This script adds a small CoELA-like layer on top of iTHOR:

    natural language goal
        -> prompt-formatted planning problem
        -> JSON task allocation plan
        -> two iTHOR agents execute assigned object pickup subtasks

It is intentionally small and presentation-oriented. It is not the original
CoELA algorithm, but it demonstrates the missing middle step between:

1. "iTHOR multi-agent works", and
2. "CoELA-style task decomposition / agent assignment".

The default planner is a deterministic prompt-compatible fallback so the demo
works without an API key. The prompt and JSON plan are saved so you can show the
"prompt engineering" artifact in the middle presentation.
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
GREEN: Color = (80, 220, 120)
WHITE: Color = (245, 245, 245)
BLACK: Color = (0, 0, 0)
DARK: Color = (20, 24, 32)


PREFERRED_OBJECT_TYPES = [
    "Apple",
    "Mug",
    "Cup",
    "Bowl",
    "Plate",
    "Bread",
    "Tomato",
    "Potato",
    "Lettuce",
    "Bottle",
    "Knife",
    "Fork",
    "Spoon",
]


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


FONT_16 = load_font(16)
FONT_18 = load_font(18)
FONT_22 = load_font(22)
FONT_28 = load_font(28)
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


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


def get_objects(event: Any) -> List[Dict[str, Any]]:
    return list(get_agent_event(event, 0).metadata.get("objects", []))


def get_agent_position(event: Any, agent_id: int) -> Dict[str, float]:
    return dict(get_agent_event(event, agent_id).metadata["agent"]["position"])


def yaw_toward(src: Dict[str, float], dst: Dict[str, float]) -> float:
    dx = float(dst["x"]) - float(src["x"])
    dz = float(dst["z"]) - float(src["z"])
    if abs(dx) + abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dz))


def dist_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((float(a["x"]) - float(b["x"])) ** 2 + (float(a["z"]) - float(b["z"])) ** 2)


def nearest_reachable(target: Dict[str, float], reachable: List[Dict[str, float]]) -> Dict[str, float]:
    return dict(min(reachable, key=lambda p: dist_xz(p, target)))


def world_to_top_pixel(
    pos: Dict[str, float],
    map_camera: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int]:
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


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
    fill: Color,
    line_spacing: int = 4,
) -> None:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_spacing
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def summarize_candidates(objects: List[Dict[str, Any]], limit: int = 24) -> List[Dict[str, Any]]:
    preferred = []
    fallback = []
    for obj in objects:
        if not obj.get("pickupable", False):
            continue
        if obj.get("isPickedUp", False):
            continue
        obj_type = obj.get("objectType", "")
        pos = obj.get("position", {})
        item = {
            "objectId": obj.get("objectId"),
            "objectType": obj_type,
            "name": obj.get("name", obj_type),
            "position": {
                "x": round(float(pos.get("x", 0.0)), 3),
                "y": round(float(pos.get("y", 0.0)), 3),
                "z": round(float(pos.get("z", 0.0)), 3),
            },
            "visible": bool(obj.get("visible", False)),
        }
        if obj_type in PREFERRED_OBJECT_TYPES:
            preferred.append(item)
        fallback.append(item)

    type_rank = {name: i for i, name in enumerate(PREFERRED_OBJECT_TYPES)}
    preferred.sort(key=lambda o: (type_rank.get(o["objectType"], 999), o["objectId"] or ""))
    fallback.sort(key=lambda o: (type_rank.get(o["objectType"], 999), o["objectType"], o["objectId"] or ""))
    candidates = preferred if len(preferred) >= 2 else fallback
    return candidates[:limit]


def build_prompt(task: str, candidates: List[Dict[str, Any]]) -> str:
    candidate_lines = []
    for i, obj in enumerate(candidates):
        pos = obj["position"]
        candidate_lines.append(
            f"{i}. objectId={obj['objectId']} | type={obj['objectType']} | "
            f"position=({pos['x']}, {pos['y']}, {pos['z']}) | visible={obj['visible']}"
        )

    return f"""You are a CoELA-style coordinator for two embodied agents in AI2-THOR.

High-level user goal:
{task}

Available agents:
- Agent 0
- Agent 1

Candidate pickupable objects:
{chr(10).join(candidate_lines)}

Allowed primitive actions:
- navigate_to_object
- pickup_object

Planning rules:
1. Decompose the high-level goal into exactly two subtasks.
2. Assign one subtask to Agent 0 and one subtask to Agent 1.
3. Use only objectIds from the candidate list.
4. Prefer assigning different object types.
5. Minimize interference by giving different target objects to the two agents.
6. Return JSON only.

Required JSON schema:
{{
  "goal": "...",
  "planner_type": "prompt_engineered_json_planner",
  "reasoning_summary": "...",
  "assignments": [
    {{
      "agent_id": 0,
      "role": "...",
      "target_object_type": "...",
      "target_object_id": "...",
      "subtask": "...",
      "actions": ["navigate_to_object", "pickup_object"]
    }},
    {{
      "agent_id": 1,
      "role": "...",
      "target_object_type": "...",
      "target_object_id": "...",
      "subtask": "...",
      "actions": ["navigate_to_object", "pickup_object"]
    }}
  ]
}}
"""


def task_mentions(task: str, obj_type: str) -> bool:
    text = task.lower()
    lower = obj_type.lower()
    if lower in text:
        return True
    aliases = {
        "Mug": ["mug", "cup"],
        "Cup": ["cup", "mug"],
        "Apple": ["apple", "fruit"],
        "Bread": ["bread", "food"],
        "Tomato": ["tomato", "food"],
        "Potato": ["potato", "food"],
        "Lettuce": ["lettuce", "food"],
        "Bottle": ["bottle", "drink"],
    }
    return any(alias in text for alias in aliases.get(obj_type, []))


def heuristic_prompt_planner(task: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic planner that follows the prompt schema.

    This keeps the demo runnable without an LLM API key while preserving the
    prompt -> JSON plan -> executor structure.
    """
    if len(candidates) < 2:
        raise RuntimeError("Need at least two pickupable candidate objects for the demo.")

    mentioned = [obj for obj in candidates if task_mentions(task, obj["objectType"])]
    pool = mentioned + [obj for obj in candidates if obj not in mentioned]

    selected: List[Dict[str, Any]] = []
    used_types = set()
    for obj in pool:
        if obj["objectType"] in used_types:
            continue
        selected.append(obj)
        used_types.add(obj["objectType"])
        if len(selected) == 2:
            break

    if len(selected) < 2:
        for obj in pool:
            if obj not in selected:
                selected.append(obj)
            if len(selected) == 2:
                break

    # Spatially assign left object to Agent 0 and right object to Agent 1 when possible.
    selected = sorted(selected, key=lambda o: o["position"]["x"])

    assignments = []
    for agent_id, obj in enumerate(selected):
        assignments.append(
            {
                "agent_id": agent_id,
                "role": "object collector",
                "target_object_type": obj["objectType"],
                "target_object_id": obj["objectId"],
                "subtask": f"Navigate to the {obj['objectType']} and pick it up.",
                "actions": ["navigate_to_object", "pickup_object"],
            }
        )

    return {
        "goal": task,
        "planner_type": "prompt_engineered_json_planner_with_deterministic_fallback",
        "reasoning_summary": (
            "The goal was decomposed into two pickup subtasks. Each agent receives "
            "a different target object to reduce interference."
        ),
        "assignments": assignments,
    }


def find_object_by_id(objects: Iterable[Dict[str, Any]], object_id: str) -> Dict[str, Any]:
    for obj in objects:
        if obj.get("objectId") == object_id:
            return obj
    raise KeyError(f"Object not found: {object_id}")


def save_composite(
    event: Any,
    frame_path: Path,
    map_camera: Dict[str, Any],
    plan: Dict[str, Any],
    caption: str,
    status_lines: List[str],
) -> None:
    canvas_w, canvas_h = 1280, 720
    top_h = 500
    bottom_h = canvas_h - top_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), BLACK)
    draw = ImageDraw.Draw(canvas)

    top_arr = get_top_frame(event)
    if top_arr is None:
        top_arr = get_agent_event(event, 0).frame
    top_img = Image.fromarray(top_arr).convert("RGB").resize((canvas_w, top_h), RESAMPLE_LANCZOS)
    canvas.paste(top_img, (0, 0))

    draw.rectangle((0, 0, canvas_w, 76), fill=BLACK)
    draw.text((20, 12), "Prompt-engineered CoELA-style planner on iTHOR", fill=YELLOW, font=FONT_28)
    draw.text((20, 45), caption, fill=WHITE, font=FONT_18)

    draw_box(draw, (765, 88, 1260, 280), DARK)
    draw.text((785, 104), "Planner output", fill=YELLOW, font=FONT_22)

    y = 136
    colors = {0: RED, 1: BLUE}
    for assignment in plan["assignments"]:
        agent_id = int(assignment["agent_id"])
        line1 = f"A{agent_id}: {assignment['target_object_type']}"
        line2 = assignment["subtask"]
        draw.text((785, y), line1, fill=colors.get(agent_id, WHITE), font=FONT_18)
        draw.text((785, y + 24), line2[:52], fill=WHITE, font=FONT_16)
        y += 58

    draw_box(draw, (20, 88, 500, 190), DARK)
    draw.text((38, 104), "Natural-language goal", fill=YELLOW, font=FONT_22)
    goal = str(plan.get("goal", ""))
    draw_wrapped_text(draw, goal, (38, 136), 430, FONT_18, WHITE)

    for agent_id, color, label in [(0, RED, "A0"), (1, BLUE, "A1")]:
        pos = get_agent_position(event, agent_id)
        px, py = world_to_top_pixel(pos, map_camera, (canvas_w, top_h))
        draw_marker(draw, (px, py), color, label)

    # Status strip.
    draw.rectangle((0, top_h - 48, canvas_w, top_h), fill=BLACK)
    x = 20
    for line in status_lines[:3]:
        draw.text((x, top_h - 34), line, fill=GREEN if "success" in line.lower() else WHITE, font=FONT_18)
        x += 400

    # Bottom agent views.
    for agent_id, x0, color in [(0, 0, RED), (1, canvas_w // 2, BLUE)]:
        ev = get_agent_event(event, agent_id)
        view = Image.fromarray(ev.frame).convert("RGB").resize((canvas_w // 2, bottom_h), RESAMPLE_LANCZOS)
        canvas.paste(view, (x0, top_h))
        draw.rectangle((x0, top_h, x0 + canvas_w // 2, top_h + 42), fill=BLACK)
        draw.text((x0 + 18, top_h + 10), f"Agent {agent_id} view", fill=color, font=FONT_22)

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
    parser.add_argument(
        "--task",
        default="Find and pick up two useful kitchen objects. Prefer an apple and a mug if available.",
    )
    parser.add_argument("--output", default="ithor_prompt_output")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output).expanduser().resolve()
    frame_dir = out_dir / "frames"
    video_path = out_dir / "ithor_prompt_planner_demo.mp4"
    prompt_path = out_dir / "planner_prompt.txt"
    plan_path = out_dir / "planner_output.json"
    log_path = out_dir / "execution_log.json"
    objects_path = out_dir / "scene_pickupable_objects.json"

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    controller = Controller(
        scene=args.scene,
        agentCount=2,
        width=args.width,
        height=args.height,
        visibilityDistance=2.0,
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    execution_log: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]

        event = controller.step(action="GetMapViewCameraProperties")
        map_camera = dict(event.metadata["actionReturn"])
        map_camera["orthographicSize"] = float(map_camera.get("orthographicSize", 5.0)) * 1.08
        event = controller.step(action="AddThirdPartyCamera", **map_camera)

        objects = get_objects(event)
        candidates = summarize_candidates(objects)
        objects_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        print(f"[info] pickupable candidates: {len(candidates)}")
        if candidates:
            print("[info] first candidates:", ", ".join(f"{o['objectType']}:{o['objectId']}" for o in candidates[:5]))
        prompt = build_prompt(args.task, candidates)
        prompt_path.write_text(prompt, encoding="utf-8")

        plan = heuristic_prompt_planner(args.task, candidates)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        for _ in range(args.hold_frames):
            save_composite(
                event,
                frame_dir / f"frame_{frame_idx:04d}.jpg",
                map_camera,
                plan,
                "Step 1/4: natural-language goal is converted into a constrained JSON planning prompt.",
                ["prompt saved", "JSON plan generated"],
            )
            frame_idx += 1

        # Execute each assignment. We do it sequentially for reliability, but the
        # plan is still two-agent task allocation.
        for assignment in plan["assignments"]:
            agent_id = int(assignment["agent_id"])
            target_id = assignment["target_object_id"]
            current_objects = get_objects(event)
            target = find_object_by_id(current_objects, target_id)
            target_pos = dict(target["position"])
            nav_pos = nearest_reachable(target_pos, reachable)
            yaw = yaw_toward(nav_pos, target_pos)

            event = controller.step(
                action="TeleportFull",
                agentId=agent_id,
                x=nav_pos["x"],
                y=nav_pos["y"],
                z=nav_pos["z"],
                rotation={"x": 0, "y": yaw, "z": 0},
                horizon=30,
                standing=True,
                forceAction=False,
            )
            nav_success = bool(get_agent_event(event, agent_id).metadata.get("lastActionSuccess", False))
            execution_log.append(
                {
                    "agent_id": agent_id,
                    "action": "navigate_to_object",
                    "target_object_id": target_id,
                    "success": nav_success,
                    "agent_position": nav_pos,
                }
            )

            for _ in range(args.hold_frames):
                save_composite(
                    event,
                    frame_dir / f"frame_{frame_idx:04d}.jpg",
                    map_camera,
                    plan,
                    f"Step 2/4: Agent {agent_id} navigates to assigned target: {target['objectType']}.",
                    [f"A{agent_id} navigate {'success' if nav_success else 'failed'}"],
                )
                frame_idx += 1

            event = controller.step(
                action="PickupObject",
                agentId=agent_id,
                objectId=target_id,
                forceAction=True,
                manualInteract=False,
            )
            pickup_success = bool(get_agent_event(event, agent_id).metadata.get("lastActionSuccess", False))
            error_message = get_agent_event(event, agent_id).metadata.get("errorMessage", "")
            execution_log.append(
                {
                    "agent_id": agent_id,
                    "action": "pickup_object",
                    "target_object_id": target_id,
                    "success": pickup_success,
                    "errorMessage": error_message,
                }
            )

            for _ in range(args.hold_frames):
                save_composite(
                    event,
                    frame_dir / f"frame_{frame_idx:04d}.jpg",
                    map_camera,
                    plan,
                    f"Step 3/4: Agent {agent_id} executes pickup_object on {target['objectType']}.",
                    [f"A{agent_id} pickup {'success' if pickup_success else 'failed'}"],
                )
                frame_idx += 1

        log_path.write_text(json.dumps(execution_log, indent=2), encoding="utf-8")

        for _ in range(args.hold_frames):
            save_composite(
                event,
                frame_dir / f"frame_{frame_idx:04d}.jpg",
                map_camera,
                plan,
                "Step 4/4: execution trace, planner prompt, and JSON plan are saved for analysis.",
                ["demo complete", f"frames={frame_idx}"],
            )
            frame_idx += 1

        video_ok = create_video(frame_dir, video_path, args.fps)
        print(f"[ok] prompt: {prompt_path}")
        print(f"[ok] planner json: {plan_path}")
        print(f"[ok] execution log: {log_path}")
        print(f"[ok] frames: {frame_dir}")
        if video_ok:
            print(f"[ok] video: {video_path}")
        else:
            print("[warn] imageio/imageio-ffmpeg not available; frames were saved.")
            print("       Install with: python -m pip install imageio imageio-ffmpeg")
            print(f"       Or create a video from: {frame_dir}/frame_%04d.jpg")

    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
