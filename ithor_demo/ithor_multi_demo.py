from pathlib import Path
from PIL import Image, ImageDraw
from ai2thor.controller import Controller

out = Path("ithor_output/frames")
out.mkdir(parents=True, exist_ok=True)

controller = Controller(
    scene="FloorPlan1",
    agentMode="default",
    agentCount=2,
    width=800,
    height=600,
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,
)

# reachable positions 가져오기
event = controller.step(action="GetReachablePositions")
positions = event.metadata["actionReturn"]

p0 = positions[len(positions) // 3]
p1 = positions[(len(positions) * 2) // 3]

# 두 agent를 보기 좋은 위치로 강제 배치
controller.step(
    action="TeleportFull",
    agentId=0,
    x=p0["x"],
    y=p0["y"],
    z=p0["z"],
    rotation={"x": 0, "y": 0, "z": 0},
    horizon=30,
    standing=True,
)

controller.step(
    action="TeleportFull",
    agentId=1,
    x=p1["x"],
    y=p1["y"],
    z=p1["z"],
    rotation={"x": 0, "y": 180, "z": 0},
    horizon=30,
    standing=True,
)

# 발표용 top-dwn 카메라
event = controller.step(
    action="AddThirdPartyCamera",
    position={"x": 0, "y": 4.5, "z": 0},
    rotation={"x": 90, "y": 0, "z": 0},
    fieldOfView=90,
    orthographic=True,
    orthographicSize=4.5,
)

def get_top_frame(event):
    for ev in event.events:
        if getattr(ev, "third_party_camera_frames", None):
            if len(ev.third_party_camera_frames) > 0:
                return Image.fromarray(ev.third_party_camera_frames[0]).resize((800, 600))
    return None

def save_frame(event, idx, caption):
    evs = event.events

    top = get_top_frame(event)
    a0 = Image.fromarray(evs[0].frame).resize((400, 300))
    a1 = Image.fromarray(evs[1].frame).resize((400, 300))

    canvas = Image.new("RGB", (800, 950), "black")
    draw = ImageDraw.Draw(canvas)

    if top:
        canvas.paste(top, (0, 0))

    canvas.paste(a0, (0, 630))
    canvas.paste(a1, (400, 630))

    draw.rectangle((0, 0, 800, 34), fill=(0, 0, 0))
    draw.text((12, 10), caption, fill=(255, 255, 0))
    draw.text((12, 605), "Top: third-party top-down camera", fill=(255, 255, 255))
    draw.text((12, 930), "Agent 0 view", fill=(120, 200, 255))
    draw.text((412, 930), "Agent 1 view", fill=(255, 220, 120))

    canvas.save(out / f"frame_{idx:04d}.jpg")

actions = [
    (0, "RotateRight"),
    (0, "MoveAhead"),
    (1, "RotateLeft"),
    (1, "MoveAhead"),
    (0, "RotateLeft"),
    (0, "MoveAhead"),
    (1, "RotateRight"),
    (1, "MoveAhead"),
] * 5

save_frame(event, 0, "iTHOR multi-agent demo: two agents initialized")

for i, (agent_id, action) in enumerate(actions, start=1):
    event = controller.step(action=action, agentId=agent_id)
    save_frame(event, i, f"step {i}: Agent {agent_id} -> {action}")

controller.stop()
print(f"saved frames to {out}")
