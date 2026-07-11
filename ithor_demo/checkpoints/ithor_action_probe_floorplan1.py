#!/usr/bin/env python3
"""
Probe which AI2-THOR interaction actions work in a scene.

Run this before building a longer cooking/serving scenario. It checks available
pickupable, receptacle, toggleable, fillable, and sliceable objects and then
tries a few candidate primitive actions with forceAction=True.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai2thor.controller import Controller


PREFERRED_PICKUP = ["Mug", "Cup", "Apple", "Lettuce", "Tomato", "Bread", "Potato", "Bowl", "Plate"]
PREFERRED_RECEPTACLE = ["DiningTable", "CounterTop", "Sink", "TableTop", "CoffeeTable", "Bowl", "Plate"]
PREFERRED_TOGGLE = ["Faucet", "StoveKnob", "Microwave", "CoffeeMachine", "Toaster"]


def as_events(event: Any) -> List[Any]:
    return list(getattr(event, "events", []) or [event])


def get_agent_event(event: Any, agent_id: int = 0) -> Any:
    events = as_events(event)
    return events[agent_id] if agent_id < len(events) else events[0]


def get_objects(event: Any) -> List[Dict[str, Any]]:
    return list(get_agent_event(event, 0).metadata.get("objects", []))


def dist_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((float(a["x"]) - float(b["x"])) ** 2 + (float(a["z"]) - float(b["z"])) ** 2)


def yaw_toward(src: Dict[str, float], dst: Dict[str, float]) -> float:
    dx = float(dst["x"]) - float(src["x"])
    dz = float(dst["z"]) - float(src["z"])
    if abs(dx) + abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dz))


def nearest_reachable(target: Dict[str, float], reachable: List[Dict[str, float]]) -> Dict[str, float]:
    return dict(min(reachable, key=lambda p: dist_xz(p, target)))


def rank_by_preference(obj: Dict[str, Any], prefs: List[str]) -> tuple:
    obj_type = obj.get("objectType", "")
    try:
        rank = prefs.index(obj_type)
    except ValueError:
        rank = 999
    return rank, obj_type, obj.get("objectId", "")


def compact_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "objectId",
        "objectType",
        "name",
        "visible",
        "pickupable",
        "receptacle",
        "toggleable",
        "isToggled",
        "sliceable",
        "isSliced",
        "canFillWithLiquid",
        "isFilledWithLiquid",
        "fillLiquid",
    ]
    return {k: obj.get(k) for k in keys if k in obj}


def find_first(objects: Iterable[Dict[str, Any]], predicate, prefs: List[str]) -> Optional[Dict[str, Any]]:
    matches = [obj for obj in objects if predicate(obj)]
    matches.sort(key=lambda obj: rank_by_preference(obj, prefs))
    return matches[0] if matches else None


def teleport_near(controller: Controller, event: Any, obj: Dict[str, Any], reachable: List[Dict[str, float]]) -> Any:
    pos = dict(obj["position"])
    nav = nearest_reachable(pos, reachable)
    return controller.step(
        action="TeleportFull",
        agentId=0,
        x=nav["x"],
        y=nav["y"],
        z=nav["z"],
        rotation={"x": 0, "y": yaw_toward(nav, pos), "z": 0},
        horizon=30,
        standing=True,
        forceAction=True,
    )


def last_result(event: Any) -> Dict[str, Any]:
    meta = get_agent_event(event, 0).metadata
    return {
        "success": bool(meta.get("lastActionSuccess", False)),
        "errorMessage": meta.get("errorMessage", ""),
        "lastAction": meta.get("lastAction"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--output", default="ithor_action_probe_output.json")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    controller = Controller(
        scene=args.scene,
        agentCount=1,
        width=args.width,
        height=args.height,
        visibilityDistance=2.0,
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    report: Dict[str, Any] = {"scene": args.scene, "tests": []}
    try:
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]
        event = controller.last_event
        objects = get_objects(event)

        categories = {
            "pickupable": [compact_obj(o) for o in objects if o.get("pickupable", False)],
            "receptacle": [compact_obj(o) for o in objects if o.get("receptacle", False)],
            "toggleable": [compact_obj(o) for o in objects if o.get("toggleable", False)],
            "fillable": [compact_obj(o) for o in objects if o.get("canFillWithLiquid", False)],
            "sliceable": [compact_obj(o) for o in objects if o.get("sliceable", False)],
        }
        report["categories"] = categories

        pickup = find_first(objects, lambda o: o.get("pickupable", False), PREFERRED_PICKUP)
        receptacle = find_first(objects, lambda o: o.get("receptacle", False), PREFERRED_RECEPTACLE)
        toggleable = find_first(objects, lambda o: o.get("toggleable", False), PREFERRED_TOGGLE)
        fillable = find_first(objects, lambda o: o.get("canFillWithLiquid", False), ["Mug", "Cup", "Bowl"])
        sliceable = find_first(objects, lambda o: o.get("sliceable", False), ["Lettuce", "Tomato", "Bread", "Potato", "Apple"])

        if pickup:
            event = teleport_near(controller, event, pickup, reachable)
            event = controller.step(action="PickupObject", objectId=pickup["objectId"], forceAction=True, manualInteract=False)
            report["tests"].append({"action": "PickupObject", "object": compact_obj(pickup), **last_result(event)})

        if pickup and receptacle:
            event = teleport_near(controller, event, receptacle, reachable)
            event = controller.step(action="PutObject", objectId=receptacle["objectId"], forceAction=True, placeStationary=True)
            report["tests"].append({"action": "PutObject", "receptacle": compact_obj(receptacle), **last_result(event)})

        if toggleable:
            event = teleport_near(controller, event, toggleable, reachable)
            event = controller.step(action="ToggleObjectOn", objectId=toggleable["objectId"], forceAction=True)
            report["tests"].append({"action": "ToggleObjectOn", "object": compact_obj(toggleable), **last_result(event)})
            event = controller.step(action="ToggleObjectOff", objectId=toggleable["objectId"], forceAction=True)
            report["tests"].append({"action": "ToggleObjectOff", "object": compact_obj(toggleable), **last_result(event)})

        if fillable:
            event = teleport_near(controller, event, fillable, reachable)
            event = controller.step(action="FillObjectWithLiquid", objectId=fillable["objectId"], fillLiquid="water", forceAction=True)
            report["tests"].append({"action": "FillObjectWithLiquid", "object": compact_obj(fillable), **last_result(event)})

        if sliceable:
            event = teleport_near(controller, event, sliceable, reachable)
            event = controller.step(action="SliceObject", objectId=sliceable["objectId"], forceAction=True)
            report["tests"].append({"action": "SliceObject", "object": compact_obj(sliceable), **last_result(event)})

        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["tests"], indent=2, ensure_ascii=False))
        print(f"[ok] wrote {Path(args.output).resolve()}")

    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
