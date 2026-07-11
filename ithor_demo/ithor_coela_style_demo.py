import json
from pathlib import Path
from PIL import Image, ImageDraw
from ai2thor.controller import Controller

out = Path("ithor_coela_style_output")
frames = out / "frames"
frames.mkdir(parents=True, exist_ok=True)

command = "Explore the room with two robots and inspect different areas."
plan = {
    "command": command,
    "task_decomposition": [
        "Robot 0 explores the left-side reachable area.",
        "Robot 1 explores the right-side reachable area.",
        "Both agents report observations through action logs."
    ],
    "assignment": {
        "agent_0": "left-area exploration",
        "agent_1": "right-area exploration"
    }
}

controller = Controller(
    scene="FloorPlan1",
    width=800,
    height=600,
    agentMode="default",
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,
)

event = controller.step(
    action="Initialize",
    agentMode="default",
    agentCount=2,
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,
    visibilityDistance=1,
    fieldOfView=90,
    makeAgentsVisible=True,
)

event = controller.step(action="GetMapViewCameraProperties")
cam = event.metadata["actionReturn"]
event = controller.step(action="AddThirdPartyCamera", **cam)

positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
cx = sum(p["x"] for p in positions) / len(positions)
cz = sum(p["z"] for p in positions) / len(positions)

def nearest(target_x, target_z):
    return min(positions, key=lambda p: (p["x"] - target_x) ** 2 + (p["z"] - target_z) ** 2)

min_x, max_x = min(p["x"] for p in positions), max(p["x"] for p in positions)
min_z, max_z = min(p["z"] for p in positions), max(p["z"] for p in positions)

agent0_path = [
    nearest(cx, cz),
    nearest(min_x, cz),
    nearest(min_x, min_z),
    nearest(cx, min_z),
    nearest(cx, cz),
]

agent1_path = [
    nearest(cx, cz),
    nearest(max_x, cz),
    nearest(max_x, max_z),
    nearest(cx, max_z),
    nearest(cx, cz),
]

def teleport(agent_id, p, yaw):
    return controller.step(
        action="TeleportFull",
        agentId=agent_id,
        x=p["x"],
        y=p["y"],
        z=p["z"],
        rotation={"x": 0, "y": yaw, "z": 0},
        horizon=30,
        standing=True,
        forceAction=True,
    )

event = teleport(0, agent0_path[0], 0)
event = teleport(1, agent1_path[0], 180)

def top_frame(event):
    return Image.fromarray(event.events[0].third_party_camera_frames[-1]).resize((800, 600))

def save_frame(event, idx, caption):
    evs = event.events
    top = top_frame(event)
    a0 = Image.fromarray(evs[0].frame).resize((400, 300))
    a1 = Image.fromarray(evs[1].frame).resize((400, 300))

    canvas = Image.new("RGB", (800, 960), "black")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(top, (0, 40))
    canvas.paste(a0, (0, 660))
    canvas.paste(a1, (400, 660))

    draw.rectangle((0, 0, 800, 40), fill=(0, 0, 0))
    draw.text((12, 12), caption, fill=(255, 255, 0))
    draw.text((12, 625), "Top: iTHOR map-view camera / Bottom: each agent view", fill=(255, 255, 255))
    draw.text((12, 940), "Agent 0: left-area exploration", fill=(120, 200, 255))
    draw.text((412, 940), "Agent 1: right-area exploration", fill=(255, 220, 120))

    canvas.save(frames / f"frame_{idx:04d}.jpg")

log = []
save_frame(event, 0, "CoELA-style iTHOR demo: task decomposition and assignment")

idx = 1
for step_i in range(1, len(agent0_path)):
    event = teleport(0, agent0_path[step_i], 90)
    log.append({"step": idx, "agent": 0, "task": plan["assignment"]["agent_0"], "position": agent0_path[step_i]})
    save_frame(event, idx, f"step {idx}: Agent 0 executes assigned left-area waypoint")
    idx += 1

    event = teleport(1, agent1_path[step_i], 270)
    log.append({"step": idx, "agent": 1, "task": plan["assignment"]["agent_1"], "position": agent1_path[step_i]})
    save_frame(event, idx, f"step {idx}: Agent 1 executes assigned right-area waypoint")
    idx += 1

plan["execution_log"] = log
(out / "plan_log.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

controller.stop()
print(f"saved frames to {frames}")
print(f"saved plan log to {out / 'plan_log.json'}")
