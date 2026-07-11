#!/usr/bin/env python3
"""
CoELA-Lite on AI2-THOR/iTHOR.

This script implements the five CoELA-style modules explicitly:

    Perception -> Memory -> Communication -> Planning -> Execution

It is not a drop-in port of the original CoELA repository because the original
paper code targets TDW-MAT / C-WAH, not AI2-THOR. Instead, this is an
AI2-THOR implementation of the CoELA modular idea:

    - each agent has its own memory,
    - each agent produces a natural-language message,
    - each agent plans using its own memory + shared message history,
    - each agent executes an object pickup-and-delivery subtask in AI2-THOR.

The script can use a local Ollama model for communication/planning. If Ollama is
not available or returns invalid JSON, it falls back to deterministic heuristics
so the final demo remains reliable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
DARK: Color = (18, 24, 34)
GREY: Color = (115, 125, 140)


PREFERRED_OBJECT_TYPES = [
    "Mug",
    "Apple",
    "Cup",
    "Bowl",
    "Plate",
    "Bread",
    "Tomato",
    "Potato",
    "Lettuce",
    "Bottle",
]

FILLABLE_OBJECT_TYPES = ["Bowl", "Mug", "Cup"]
SLICEABLE_OBJECT_TYPES = ["Lettuce", "Tomato", "Potato", "Bread", "Apple"]
TOGGLE_OBJECT_TYPES = ["Faucet", "StoveKnob", "Microwave", "CoffeeMachine", "Toaster"]

PREFERRED_RECEPTACLE_TYPES = [
    "CounterTop",
    "DiningTable",
    "TableTop",
    "CoffeeTable",
    "Sink",
    "Bowl",
    "Plate",
]

# For this demo, delivery targets should be open surfaces. AI2-THOR can model
# containers such as Fridge/Cabinet, but those often require extra OpenObject /
# CloseObject state transitions. Keeping the delivery destination to open
# surfaces makes the cooperative task reliable and easy to explain.
SAFE_DELIVERY_RECEPTACLE_TYPES = set(PREFERRED_RECEPTACLE_TYPES)

TRANSPORT_GOAL = (
    "Find two useful kitchen objects, coordinate so each agent handles a different object, "
    "then deliver both objects to a shared receptacle."
)

KITCHEN_PREP_GOAL = (
    "Coordinate a longer kitchen-preparation task. Agent 0 should fill a bowl, mug, or cup with "
    "water using the faucet and deliver it to a countertop. Agent 1 should slice a food "
    "object such as lettuce, tomato, potato, bread, or apple, then deliver it to an "
    "available open countertop."
)


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


FONT_14 = load_font(14)
FONT_16 = load_font(16)
FONT_18 = load_font(18)
FONT_20 = load_font(20)
FONT_22 = load_font(22)
FONT_24 = load_font(24)
FONT_26 = load_font(26)
FONT_28 = load_font(28)
FONT_32 = load_font(32)
FONT_36 = load_font(36)
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


def get_agent_position(event: Any, agent_id: int) -> Dict[str, float]:
    return dict(get_agent_event(event, agent_id).metadata["agent"]["position"])


def get_all_objects(event: Any) -> List[Dict[str, Any]]:
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


def viewing_reachable(
    target: Dict[str, float],
    reachable: List[Dict[str, float]],
    preferred_distance: float = 1.1,
) -> Dict[str, float]:
    """Choose a reachable point that is close enough to see the target but not too close.

    The nearest reachable position can put the agent almost on top of a small cup,
    which often pushes the delivered object out of the egocentric camera view.
    """
    return dict(
        min(
            reachable,
            key=lambda p: (
                abs(dist_xz(p, target) - preferred_distance),
                dist_xz(p, target),
            ),
        )
    )


def object_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    pos = obj.get("position", {})
    return {
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "name": obj.get("name", obj.get("objectType")),
        "pickupable": bool(obj.get("pickupable", False)),
        "receptacle": bool(obj.get("receptacle", False)),
        "visible": bool(obj.get("visible", False)),
        "isPickedUp": bool(obj.get("isPickedUp", False)),
        "toggleable": bool(obj.get("toggleable", False)),
        "isToggled": bool(obj.get("isToggled", False)),
        "sliceable": bool(obj.get("sliceable", False)),
        "isSliced": bool(obj.get("isSliced", False)),
        "canFillWithLiquid": bool(obj.get("canFillWithLiquid", False)),
        "isFilledWithLiquid": bool(obj.get("isFilledWithLiquid", False)),
        "fillLiquid": obj.get("fillLiquid"),
        "parentReceptacles": list(obj.get("parentReceptacles") or []),
        "position": {
            "x": round(float(pos.get("x", 0.0)), 3),
            "y": round(float(pos.get("y", 0.0)), 3),
            "z": round(float(pos.get("z", 0.0)), 3),
        },
    }


def type_rank(obj_type: str) -> int:
    try:
        return PREFERRED_OBJECT_TYPES.index(obj_type)
    except ValueError:
        return 999


def receptacle_rank(obj_type: str) -> int:
    try:
        return PREFERRED_RECEPTACLE_TYPES.index(obj_type)
    except ValueError:
        return 999


def extract_json(text: str) -> Dict[str, Any]:
    """Parse raw LLM text into JSON as defensively as possible."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
    return json.loads(match.group(0))


def call_ollama_json(model: str, prompt: str, timeout: int = 180) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a CoELA-style embodied-agent module. "
                        "Return valid JSON only. No markdown, no prose, no hidden reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": "/no_think\n" + prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            return extract_json(content)
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("Ollama JSON call failed")


@dataclass
class AgentPlan:
    agent_id: int
    target_object_id: Optional[str]
    target_object_type: Optional[str]
    destination_receptacle_id: Optional[str]
    destination_receptacle_type: Optional[str]
    high_level_plan: str
    primitive_actions: List[str]
    source: str


@dataclass
class CoELAAgent:
    agent_id: int
    name: str
    color_name: str
    scenario: str = "transport"
    use_llm: bool = False
    llm_model: str = "qwen3:8b"
    known_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    known_receptacles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    known_interactables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    held_objects: List[str] = field(default_factory=list)
    current_plan: Optional[AgentPlan] = None
    prompt_records: List[Dict[str, Any]] = field(default_factory=list)

    def perceive(self, event: Any, fallback_global_objects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Perception module: extract pickupable visible objects from this agent's observation."""
        ev = get_agent_event(event, self.agent_id)
        objects = list(ev.metadata.get("objects", []))
        visible_pickupables = [
            object_summary(obj)
            for obj in objects
            if obj.get("pickupable", False) and obj.get("visible", False) and not obj.get("isPickedUp", False)
        ]
        visible_receptacles = [
            object_summary(obj)
            for obj in objects
            if obj.get("receptacle", False) and obj.get("visible", False)
        ]
        visible_interactables = [
            object_summary(obj)
            for obj in objects
            if obj.get("visible", False) and (obj.get("toggleable", False) or obj.get("openable", False))
        ]

        # A robust fallback keeps the demo from failing if the initial camera pose
        # misses small objects. The fallback is recorded in the observation.
        used_fallback = False
        if not visible_pickupables and fallback_global_objects is not None:
            visible_pickupables = sorted(
                [
                    object_summary(obj)
                    for obj in fallback_global_objects
                    if obj.get("pickupable", False) and not obj.get("isPickedUp", False)
                ],
                key=lambda o: (type_rank(o["objectType"]), o["objectType"], o["objectId"] or ""),
            )[:12]
            used_fallback = True

        if not visible_interactables and fallback_global_objects is not None:
            visible_interactables = [
                object_summary(obj)
                for obj in fallback_global_objects
                if obj.get("toggleable", False) or obj.get("openable", False)
            ][:8]

        if not visible_receptacles and fallback_global_objects is not None:
            visible_receptacles = [
                object_summary(obj)
                for obj in fallback_global_objects
                if obj.get("receptacle", False) and obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
            ][:8]

        # Even if an agent can see some receptacles, it might initially see only
        # closed appliances such as a Fridge or Cabinet. Add globally known safe
        # open surfaces so both agents can coordinate on the same executable
        # delivery target.
        if fallback_global_objects is not None:
            merged_receptacles = {
                obj["objectId"]: obj
                for obj in visible_receptacles
                if obj.get("objectId") and obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
            }
            for obj in fallback_global_objects:
                if obj.get("receptacle", False) and obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES:
                    summary = object_summary(obj)
                    if summary.get("objectId"):
                        merged_receptacles.setdefault(summary["objectId"], summary)
            visible_receptacles = list(merged_receptacles.values())[:8]

        held = []
        for obj in objects:
            if obj.get("isPickedUp", False) and obj.get("objectId"):
                held.append(obj["objectId"])

        observation = {
            "agent_id": self.agent_id,
            "visible_pickupable_objects": sorted(
                visible_pickupables,
                key=lambda o: (type_rank(o["objectType"]), o["objectType"], o["objectId"] or ""),
            ),
            "visible_receptacles": sorted(
                visible_receptacles,
                key=lambda o: (receptacle_rank(o["objectType"]), o["objectType"], o["objectId"] or ""),
            ),
            "visible_interactables": sorted(
                visible_interactables,
                key=lambda o: (
                    TOGGLE_OBJECT_TYPES.index(o["objectType"])
                    if o["objectType"] in TOGGLE_OBJECT_TYPES
                    else 999,
                    o["objectType"],
                    o["objectId"] or "",
                ),
            ),
            "held_objects": held,
            "used_global_metadata_fallback": used_fallback,
        }
        return observation

    def update_memory(self, observation: Dict[str, Any]) -> None:
        """Memory module: store objects and own held objects."""
        for obj in observation["visible_pickupable_objects"]:
            if obj.get("objectId"):
                obj = dict(obj)
                obj["last_seen_by"] = self.agent_id
                obj["last_seen_time"] = time.time()
                self.known_objects[obj["objectId"]] = obj
        for obj in observation.get("visible_receptacles", []):
            if obj.get("objectId"):
                obj = dict(obj)
                obj["last_seen_by"] = self.agent_id
                obj["last_seen_time"] = time.time()
                self.known_receptacles[obj["objectId"]] = obj
        for obj in observation.get("visible_interactables", []):
            if obj.get("objectId"):
                obj = dict(obj)
                obj["last_seen_by"] = self.agent_id
                obj["last_seen_time"] = time.time()
                self.known_interactables[obj["objectId"]] = obj
        self.held_objects = list(observation.get("held_objects", []))

    def receive_messages(self, messages: List[Dict[str, Any]]) -> None:
        seen = {(m.get("from"), m.get("round"), m.get("message")) for m in self.message_history}
        for msg in messages:
            key = (msg.get("from"), msg.get("round"), msg.get("message"))
            if key not in seen:
                self.message_history.append(dict(msg))
                seen.add(key)

    def _claimed_object_ids(self) -> set:
        claimed = set()
        for msg in self.message_history:
            target = msg.get("target_object_id")
            if target:
                claimed.add(target)
        if self.current_plan and self.current_plan.target_object_id:
            claimed.add(self.current_plan.target_object_id)
        return claimed

    def _claimed_receptacle_ids_by_others(self) -> set:
        claimed = set()
        for msg in self.message_history:
            if msg.get("agent_id") == self.agent_id:
                continue
            receptacle = msg.get("destination_receptacle_id")
            if receptacle:
                claimed.add(receptacle)
        return claimed

    def _candidate_objects(self) -> List[Dict[str, Any]]:
        candidates = [
            obj
            for obj in self.known_objects.values()
            if obj.get("pickupable", True) and not obj.get("isPickedUp", False)
        ]
        candidates.sort(key=lambda o: (type_rank(o["objectType"]), o["objectType"], o["objectId"] or ""))
        return candidates

    def _candidate_receptacles(self) -> List[Dict[str, Any]]:
        candidates = [
            obj
            for obj in self.known_receptacles.values()
            if obj.get("receptacle", False) and obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
        ]

        def known_receptacle_load(receptacle: Dict[str, Any]) -> int:
            rid = receptacle.get("objectId")
            if not rid:
                return 999
            return sum(
                1
                for obj in self.known_objects.values()
                if rid in (obj.get("parentReceptacles") or [])
            )

        candidates.sort(
            key=lambda o: (
                known_receptacle_load(o),
                receptacle_rank(o["objectType"]),
                dist_xz(o.get("position", {"x": 0.0, "z": 0.0}), {"x": 0.0, "z": 0.0}),
                o["objectType"],
                o["objectId"] or "",
            )
        )
        return candidates

    def _preferred_delivery_receptacle(self, exclude_claimed_by_others: bool = False) -> Optional[Dict[str, Any]]:
        candidates = self._candidate_receptacles()
        if exclude_claimed_by_others:
            claimed = self._claimed_receptacle_ids_by_others()
            filtered = [obj for obj in candidates if obj.get("objectId") not in claimed]
            if filtered:
                candidates = filtered
        return candidates[0] if candidates else None

    def _original_parent_receptacle(self, target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not target:
            return None
        for rid in target.get("parentReceptacles") or []:
            receptacle = self.known_receptacles.get(rid)
            if (
                receptacle
                and receptacle.get("receptacle", False)
                and receptacle.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
            ):
                return receptacle
        return None

    def _sink_side_countertop_receptacle(self) -> Optional[Dict[str, Any]]:
        """Prefer a countertop near the faucet for the filled bowl/cup result.

        This keeps Agent 0's water task visually separated from Agent 1's sliced
        food placement on the island/plate area.
        """
        countertops = [
            obj
            for obj in self.known_receptacles.values()
            if obj.get("receptacle", False) and obj.get("objectType") == "CounterTop"
        ]
        if not countertops:
            return None
        faucets = [
            obj
            for obj in self.known_interactables.values()
            if obj.get("objectType") == "Faucet"
        ]
        if faucets:
            faucet_pos = faucets[0].get("position", {"x": 0.0, "z": 0.0})
            countertops.sort(
                key=lambda obj: (
                    dist_xz(obj.get("position", {"x": 0.0, "z": 0.0}), faucet_pos),
                    obj.get("objectId", ""),
                )
            )
        else:
            countertops.sort(
                key=lambda obj: (
                    obj.get("position", {}).get("x", 0.0),
                    obj.get("objectId", ""),
                )
            )
        return countertops[0]

    def _role_delivery_receptacle(self, target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self.scenario == "kitchen-prep" and self.agent_id == 0:
            return self._sink_side_countertop_receptacle() or self._original_parent_receptacle(target) or self._preferred_delivery_receptacle()
        if self.scenario == "kitchen-prep" and self.agent_id == 1:
            return self._preferred_delivery_receptacle(exclude_claimed_by_others=True)
        return self._preferred_delivery_receptacle()

    def _preferred_known_object(
        self,
        preferred_types: List[str],
        required_flag: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        for obj in self.known_objects.values():
            if obj.get("objectType") not in preferred_types:
                continue
            if required_flag and not obj.get(required_flag, False):
                continue
            if obj.get("isPickedUp", False):
                continue
            candidates.append(obj)
        candidates.sort(
            key=lambda o: (
                preferred_types.index(o["objectType"]) if o["objectType"] in preferred_types else 999,
                o["objectType"],
                o["objectId"] or "",
            )
        )
        return candidates[0] if candidates else None

    def _kitchen_prep_target(self) -> Optional[Dict[str, Any]]:
        if self.agent_id == 0:
            return self._preferred_known_object(FILLABLE_OBJECT_TYPES, "canFillWithLiquid")
        return self._preferred_known_object(SLICEABLE_OBJECT_TYPES, "sliceable")

    def _apply_kitchen_prep_message_constraints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.scenario != "kitchen-prep":
            return payload

        target = self._kitchen_prep_target()
        receptacle = self._role_delivery_receptacle(target)
        if target is not None:
            payload["target_object_id"] = target["objectId"]
            payload["target_object_type"] = target["objectType"]

        if self.agent_id == 0:
            if receptacle is not None:
                payload["destination_receptacle_id"] = receptacle["objectId"]
                payload["destination_receptacle_type"] = receptacle["objectType"]
            payload["message"] = (
                f"{self.name}: I will fill the {payload.get('target_object_type', 'mug')} "
                f"with water using the faucet, then deliver it to the "
                f"{payload.get('destination_receptacle_type', 'countertop')}."
            )
        else:
            if receptacle is not None:
                payload["destination_receptacle_id"] = receptacle["objectId"]
                payload["destination_receptacle_type"] = receptacle["objectType"]
            payload["message"] = (
                f"{self.name}: I will slice the {payload.get('target_object_type', 'lettuce')} "
                f"and deliver it to the {payload.get('destination_receptacle_type', 'available open countertop')} "
                "while you handle the water."
            )

        payload["scenario_constraints_applied"] = True
        return payload

    def _apply_kitchen_prep_plan_constraints(self, plan: AgentPlan) -> AgentPlan:
        if self.scenario != "kitchen-prep":
            return plan

        target = self._kitchen_prep_target()
        receptacle = self._role_delivery_receptacle(target)
        if target is not None:
            plan.target_object_id = target["objectId"]
            plan.target_object_type = target["objectType"]

        if self.agent_id == 0:
            if receptacle is not None:
                plan.destination_receptacle_id = receptacle["objectId"]
                plan.destination_receptacle_type = receptacle["objectType"]
            plan.primitive_actions = [
                "navigate_to_object",
                "pickup_object",
                "navigate_to_interactable",
                "toggle_object_on",
                "fill_object_with_liquid",
                "toggle_object_off",
                "navigate_to_receptacle",
                "put_object",
            ]
            plan.high_level_plan = (
                f"Pick up the {plan.target_object_type}, turn on the faucet, fill it with water, "
                f"turn the faucet off, then deliver it to the {plan.destination_receptacle_type}."
            )
        else:
            if receptacle is not None:
                plan.destination_receptacle_id = receptacle["objectId"]
                plan.destination_receptacle_type = receptacle["objectType"]
            plan.primitive_actions = [
                "navigate_to_object",
                "slice_object",
                "pickup_object",
                "navigate_to_receptacle",
                "put_object",
            ]
            plan.high_level_plan = (
                f"Navigate to the {plan.target_object_type}, slice it, pick it up, "
                f"then deliver it to the {plan.destination_receptacle_type}."
            )

        if "scenario_constraints" not in plan.source:
            plan.source = f"{plan.source}+scenario_constraints"
        return plan

    def _ensure_safe_delivery_receptacle_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Keep LLM-selected delivery targets inside executable open-surface choices."""
        chosen = None
        receptacle_id = payload.get("destination_receptacle_id")
        original_receptacle_id = receptacle_id
        original_receptacle_type = payload.get("destination_receptacle_type")
        if receptacle_id and receptacle_id in self.known_receptacles:
            maybe = self.known_receptacles[receptacle_id]
            if maybe.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES:
                chosen = maybe
        if chosen is None:
            chosen = self._preferred_delivery_receptacle()
        if chosen is not None:
            payload["destination_receptacle_id"] = chosen["objectId"]
            payload["destination_receptacle_type"] = chosen["objectType"]
            if original_receptacle_id != chosen["objectId"] or original_receptacle_type != chosen["objectType"]:
                payload["delivery_receptacle_overridden"] = True
                if payload.get("message"):
                    target = payload.get("target_object_type") or "selected object"
                    payload["message"] = (
                        f"{self.name}: I will handle the {target} and deliver it to the "
                        f"{chosen['objectType']}. Please choose a different target."
                    )
        return payload

    def build_communication_prompt(self, goal: str, round_id: int) -> str:
        candidates = self._candidate_objects()
        receptacles = self._candidate_receptacles()
        interactables = list(self.known_interactables.values())
        scenario_hint = ""
        if self.scenario == "kitchen-prep":
            scenario_hint = (
                "\nScenario-specific role:\n"
                "- Agent 0 should claim a fillable Mug/Cup/Bowl, use the Faucet, fill it with water, and deliver it.\n"
                "- Agent 1 should claim a sliceable food object such as Lettuce/Tomato/Potato/Bread/Apple, slice it, and deliver it.\n"
            )
        return f"""You are {self.name}, an embodied agent in AI2-THOR.

CoELA module: Communication.

Shared goal:
{goal}
{scenario_hint}

Your memory of pickupable objects:
{json.dumps(candidates[:10], ensure_ascii=False)}

Known delivery receptacles:
{json.dumps(receptacles[:6], ensure_ascii=False)}

Known interactable objects:
{json.dumps(interactables[:6], ensure_ascii=False)}

Message history:
{json.dumps(self.message_history[-8:], ensure_ascii=False)}

Task:
Decide what short natural-language message you should send to the other agent.
You should coordinate roles, avoid selecting the same target object, and agree on a shared delivery receptacle.
Use only a known delivery receptacle from the list above. Prefer open surfaces such as CounterTop.
Do not choose closed appliances/storage such as Fridge, Cabinet, Microwave, Toaster, or CoffeeMachine.

Return JSON only:
{{
  "agent_id": {self.agent_id},
  "round": {round_id},
  "message": "...",
  "target_object_id": "... or null",
  "target_object_type": "... or null",
  "destination_receptacle_id": "... or null",
  "destination_receptacle_type": "... or null",
  "primitive_actions": ["..."],
  "confidence": 0.0
}}
"""

    def build_compact_communication_prompt(self, goal: str, round_id: int) -> str:
        """Short retry prompt for models that fail on the long context prompt."""
        target = self._kitchen_prep_target() if self.scenario == "kitchen-prep" else None
        if target is None:
            candidates = self._candidate_objects()
            target = candidates[0] if candidates else None
        receptacle = self._role_delivery_receptacle(target)
        role = (
            "fill a bowl/mug/cup with water using the faucet"
            if self.agent_id == 0 and self.scenario == "kitchen-prep"
            else "slice a food object and deliver it"
            if self.scenario == "kitchen-prep"
            else "pick up and deliver a useful object"
        )
        return f"""Return JSON only. No markdown. No explanation.

You are Agent {self.agent_id} in a two-agent AI2-THOR cooperative task.
Goal: {goal}
Your role: {role}
Target object: {json.dumps(target, ensure_ascii=False)}
Delivery receptacle: {json.dumps(receptacle, ensure_ascii=False)}

Return exactly this JSON shape:
{{
  "agent_id": {self.agent_id},
  "round": {round_id},
  "message": "I will ...",
  "target_object_id": {json.dumps(target.get("objectId") if target else None)},
  "target_object_type": {json.dumps(target.get("objectType") if target else None)},
  "destination_receptacle_id": {json.dumps(receptacle.get("objectId") if receptacle else None)},
  "destination_receptacle_type": {json.dumps(receptacle.get("objectType") if receptacle else None)},
  "primitive_actions": ["..."],
  "confidence": 0.9
}}
"""

    def heuristic_communication(self, goal: str, round_id: int) -> Dict[str, Any]:
        candidates = self._candidate_objects()
        receptacle = self._preferred_delivery_receptacle()
        claimed = self._claimed_object_ids()
        chosen = None
        for obj in candidates:
            if obj["objectId"] not in claimed:
                chosen = obj
                break
        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen is None:
            return {
                "agent_id": self.agent_id,
                "round": round_id,
                "message": f"{self.name}: I do not see a pickupable target yet. I will keep exploring.",
                "target_object_id": None,
                "target_object_type": None,
                "destination_receptacle_id": receptacle["objectId"] if receptacle else None,
                "destination_receptacle_type": receptacle["objectType"] if receptacle else None,
                "confidence": 0.2,
                "source": "heuristic",
            }

        message = (
            f"{self.name}: I can handle the {chosen['objectType']} "
            f"({chosen['objectId']}) and deliver it to the "
            f"{receptacle['objectType'] if receptacle else 'shared delivery area'}. "
            f"Please choose a different target."
        )
        return {
            "agent_id": self.agent_id,
            "round": round_id,
            "message": message,
            "target_object_id": chosen["objectId"],
            "target_object_type": chosen["objectType"],
            "destination_receptacle_id": receptacle["objectId"] if receptacle else None,
            "destination_receptacle_type": receptacle["objectType"] if receptacle else None,
            "confidence": 0.8,
            "source": "heuristic",
        }

    def communicate(self, goal: str, round_id: int) -> Dict[str, Any]:
        """Communication module: generate a natural-language coordination message."""
        prompt = self.build_communication_prompt(goal, round_id)
        self.prompt_records.append({"module": "communication", "round": round_id, "prompt": prompt})
        if self.use_llm:
            try:
                msg = call_ollama_json(self.llm_model, prompt)
                msg.setdefault("agent_id", self.agent_id)
                msg.setdefault("round", round_id)
                msg.setdefault("source", f"ollama:{self.llm_model}")
                if "message" not in msg:
                    msg["message"] = ""
                    msg["message_field_normalized"] = True
                msg = self._ensure_safe_delivery_receptacle_fields(msg)
                msg = self._apply_kitchen_prep_message_constraints(msg)
                return msg
            except Exception as exc:
                try:
                    compact_prompt = self.build_compact_communication_prompt(goal, round_id)
                    self.prompt_records.append(
                        {
                            "module": "communication_compact_retry",
                            "round": round_id,
                            "prompt": compact_prompt,
                            "retry_after": repr(exc),
                        }
                    )
                    msg = call_ollama_json(self.llm_model, compact_prompt)
                    msg.setdefault("agent_id", self.agent_id)
                    msg.setdefault("round", round_id)
                    msg["source"] = f"ollama:{self.llm_model}+compact_json_retry"
                    if "message" not in msg:
                        msg["message"] = ""
                        msg["message_field_normalized"] = True
                    msg = self._ensure_safe_delivery_receptacle_fields(msg)
                    msg = self._apply_kitchen_prep_message_constraints(msg)
                    msg["recovered_from_long_prompt_error"] = repr(exc)
                    return msg
                except Exception as retry_exc:
                    fallback = self.heuristic_communication(goal, round_id)
                    fallback["llm_error"] = repr(exc)
                    fallback["compact_retry_error"] = repr(retry_exc)
                    fallback = self._apply_kitchen_prep_message_constraints(fallback)
                    return fallback
        fallback = self.heuristic_communication(goal, round_id)
        return self._apply_kitchen_prep_message_constraints(fallback)

    def build_planning_prompt(self, goal: str, round_id: int) -> str:
        candidates = self._candidate_objects()
        receptacles = self._candidate_receptacles()
        interactables = list(self.known_interactables.values())
        scenario_hint = ""
        if self.scenario == "kitchen-prep":
            scenario_hint = (
                "\nScenario-specific role:\n"
                "- Agent 0 plan should include: navigate_to_object, pickup_object, navigate_to_interactable, "
                "toggle_object_on, fill_object_with_liquid, toggle_object_off, navigate_to_receptacle, put_object.\n"
                "- Agent 1 plan should include: navigate_to_object, slice_object, pickup_object, "
                "navigate_to_receptacle, put_object.\n"
            )
        return f"""You are {self.name}, an embodied agent in AI2-THOR.

CoELA module: Planning.

Shared goal:
{goal}
{scenario_hint}

Your memory:
{json.dumps(candidates[:10], ensure_ascii=False)}

Known delivery receptacles:
{json.dumps(receptacles[:6], ensure_ascii=False)}

Known interactable objects:
{json.dumps(interactables[:6], ensure_ascii=False)}

Message history:
{json.dumps(self.message_history[-10:], ensure_ascii=False)}

Available high-level plans:
- navigate_to_object
- pickup_object
- navigate_to_interactable
- toggle_object_on
- fill_object_with_liquid
- toggle_object_off
- slice_object
- navigate_to_receptacle
- put_object
- wait

Task:
Choose your next high-level plan. Avoid objects already claimed by another agent.
The task is long-horizon: pick up your target object and deliver it to a shared receptacle.
Use only a known delivery receptacle from the list above. Prefer open surfaces such as CounterTop.
Do not choose closed appliances/storage such as Fridge, Cabinet, Microwave, Toaster, or CoffeeMachine.

Return JSON only:
{{
  "agent_id": {self.agent_id},
  "target_object_id": "... or null",
  "target_object_type": "... or null",
  "destination_receptacle_id": "... or null",
  "destination_receptacle_type": "... or null",
  "high_level_plan": "...",
  "primitive_actions": ["navigate_to_object", "pickup_object", "navigate_to_receptacle", "put_object"],
  "reason": "..."
}}
"""

    def build_compact_planning_prompt(self, goal: str, round_id: int) -> str:
        """Short JSON-only retry prompt for planning.

        The long prompt is useful for demonstrating CoELA-style context, but
        smaller local models can occasionally fail to emit a JSON object. This
        compact retry keeps the LLM in the loop while removing irrelevant text.
        """
        target = self._kitchen_prep_target() if self.scenario == "kitchen-prep" else None
        if target is None:
            candidates = self._candidate_objects()
            target = candidates[0] if candidates else None
        receptacle = self._role_delivery_receptacle(target)

        if self.scenario == "kitchen-prep" and self.agent_id == 0:
            primitive_actions = [
                "navigate_to_object",
                "pickup_object",
                "navigate_to_interactable",
                "toggle_object_on",
                "fill_object_with_liquid",
                "toggle_object_off",
                "navigate_to_receptacle",
                "put_object",
            ]
            high_level_plan = (
                f"Pick up the {target.get('objectType') if target else 'mug'}, fill it with water using the faucet, "
                f"then deliver it to the {receptacle.get('objectType') if receptacle else 'countertop'}."
            )
        elif self.scenario == "kitchen-prep":
            primitive_actions = [
                "navigate_to_object",
                "slice_object",
                "pickup_object",
                "navigate_to_receptacle",
                "put_object",
            ]
            high_level_plan = (
                f"Slice the {target.get('objectType') if target else 'food object'}, pick it up, "
                f"then deliver it to the {receptacle.get('objectType') if receptacle else 'countertop'}."
            )
        else:
            primitive_actions = ["navigate_to_object", "pickup_object", "navigate_to_receptacle", "put_object"]
            high_level_plan = (
                f"Pick up the {target.get('objectType') if target else 'object'} and deliver it to "
                f"the {receptacle.get('objectType') if receptacle else 'shared receptacle'}."
            )

        return f"""Return JSON only. No markdown. No explanation.

You are Agent {self.agent_id} in AI2-THOR.
Goal: {goal}
Target object: {json.dumps(target, ensure_ascii=False)}
Delivery receptacle: {json.dumps(receptacle, ensure_ascii=False)}
Primitive actions: {json.dumps(primitive_actions)}

Return exactly this JSON shape:
{{
  "agent_id": {self.agent_id},
  "target_object_id": {json.dumps(target.get("objectId") if target else None)},
  "target_object_type": {json.dumps(target.get("objectType") if target else None)},
  "destination_receptacle_id": {json.dumps(receptacle.get("objectId") if receptacle else None)},
  "destination_receptacle_type": {json.dumps(receptacle.get("objectType") if receptacle else None)},
  "high_level_plan": {json.dumps(high_level_plan)},
  "primitive_actions": {json.dumps(primitive_actions)},
  "reason": "Use the assigned role and avoid interfering with the other agent."
}}
"""

    def heuristic_plan(self, goal: str) -> AgentPlan:
        candidates = self._candidate_objects()
        scenario_target = self._kitchen_prep_target() if self.scenario == "kitchen-prep" else None
        receptacle = self._role_delivery_receptacle(scenario_target)
        claimed_by_others = {
            msg.get("target_object_id")
            for msg in self.message_history
            if msg.get("agent_id") != self.agent_id and msg.get("target_object_id")
        }

        # Prefer the object this agent claimed in its own communication.
        own_claim = None
        for msg in reversed(self.message_history):
            if msg.get("agent_id") == self.agent_id and msg.get("target_object_id"):
                own_claim = msg["target_object_id"]
                break

        chosen = None
        if own_claim:
            chosen = self.known_objects.get(own_claim)
        if chosen is None:
            for obj in candidates:
                if obj["objectId"] not in claimed_by_others:
                    chosen = obj
                    break
        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen is None:
            return AgentPlan(
                agent_id=self.agent_id,
                target_object_id=None,
                target_object_type=None,
                destination_receptacle_id=receptacle["objectId"] if receptacle else None,
                destination_receptacle_type=receptacle["objectType"] if receptacle else None,
                high_level_plan="wait and observe",
                primitive_actions=["wait"],
                source="heuristic",
            )

        return AgentPlan(
            agent_id=self.agent_id,
            target_object_id=chosen["objectId"],
            target_object_type=chosen["objectType"],
            destination_receptacle_id=receptacle["objectId"] if receptacle else None,
            destination_receptacle_type=receptacle["objectType"] if receptacle else None,
            high_level_plan=(
                f"Navigate to the {chosen['objectType']}, pick it up, "
                f"and deliver it to the {receptacle['objectType'] if receptacle else 'shared receptacle'}."
            ),
            primitive_actions=["navigate_to_object", "pickup_object", "navigate_to_receptacle", "put_object"],
            source="heuristic",
        )

    def plan(self, goal: str, round_id: int) -> AgentPlan:
        """Planning module: choose a high-level plan using memory and messages."""
        prompt = self.build_planning_prompt(goal, round_id)
        self.prompt_records.append({"module": "planning", "round": round_id, "prompt": prompt})

        def normalize_raw_plan(raw: Dict[str, Any], source: str) -> AgentPlan:
            plan = AgentPlan(
                agent_id=self.agent_id,
                target_object_id=raw.get("target_object_id"),
                target_object_type=raw.get("target_object_type"),
                destination_receptacle_id=raw.get("destination_receptacle_id"),
                destination_receptacle_type=raw.get("destination_receptacle_type"),
                high_level_plan=raw.get("high_level_plan") or raw.get("plan") or "wait",
                primitive_actions=list(
                    raw.get("primitive_actions")
                    or ["navigate_to_object", "pickup_object", "navigate_to_receptacle", "put_object"]
                ),
                source=source,
            )
            if (
                plan.target_object_id
                and plan.target_object_id not in self.known_objects
                and self.scenario != "kitchen-prep"
            ):
                raise ValueError("LLM selected an object that is not in this agent's memory")
            safe_delivery = self._ensure_safe_delivery_receptacle_fields(
                {
                    "destination_receptacle_id": plan.destination_receptacle_id,
                    "destination_receptacle_type": plan.destination_receptacle_type,
                }
            )
            plan.destination_receptacle_id = safe_delivery.get("destination_receptacle_id")
            plan.destination_receptacle_type = safe_delivery.get("destination_receptacle_type")
            if plan.destination_receptacle_id and plan.destination_receptacle_id not in self.known_receptacles:
                raise ValueError("LLM selected a receptacle that is not in this agent's memory")
            return self._apply_kitchen_prep_plan_constraints(plan)

        if self.use_llm:
            try:
                raw = call_ollama_json(self.llm_model, prompt)
                plan = normalize_raw_plan(raw, f"ollama:{self.llm_model}")
                self.current_plan = plan
                return plan
            except Exception as exc:
                try:
                    compact_prompt = self.build_compact_planning_prompt(goal, round_id)
                    self.prompt_records.append(
                        {
                            "module": "planning_compact_retry",
                            "round": round_id,
                            "prompt": compact_prompt,
                            "retry_after": repr(exc),
                        }
                    )
                    raw = call_ollama_json(self.llm_model, compact_prompt)
                    plan = normalize_raw_plan(raw, f"ollama:{self.llm_model}+compact_json_retry")
                    plan.high_level_plan = f"{plan.high_level_plan} Compact JSON retry recovered from long-prompt parse failure."
                    self.current_plan = plan
                    return plan
                except Exception as retry_exc:
                    plan = self.heuristic_plan(goal)
                    plan.source = f"heuristic_fallback_after_llm_error:{repr(exc)}; compact_retry_error:{repr(retry_exc)}"
                    plan = self._apply_kitchen_prep_plan_constraints(plan)
                    self.current_plan = plan
                    return plan

        plan = self.heuristic_plan(goal)
        plan = self._apply_kitchen_prep_plan_constraints(plan)
        self.current_plan = plan
        return plan

    def memory_snapshot(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "known_objects": list(self.known_objects.values()),
            "known_receptacles": list(self.known_receptacles.values()),
            "known_interactables": list(self.known_interactables.values()),
            "held_objects": self.held_objects,
            "message_history": self.message_history,
            "action_history": self.action_history,
            "current_plan": self.current_plan.__dict__ if self.current_plan else None,
        }


def find_object_by_id(objects: Iterable[Dict[str, Any]], object_id: str) -> Dict[str, Any]:
    for obj in objects:
        if obj.get("objectId") == object_id:
            return obj
    raise KeyError(f"Object not found: {object_id}")


def find_transportable_after_slice(
    objects: Iterable[Dict[str, Any]],
    original_target: Dict[str, Any],
) -> Dict[str, Any]:
    """Find a pickupable object to transport after SliceObject.

    Some AI2-THOR objects keep the same objectId after slicing; others create
    sliced variants. Prefer a sliced variant first because smaller sliced food
    pieces are much more likely to fit on a Bowl/Plate/CounterTop.
    """
    original_id = original_target.get("objectId")
    original_type = original_target.get("objectType", "")
    original_pos = original_target.get("position", {"x": 0.0, "z": 0.0})

    object_list = list(objects)
    sliced_variants = []
    related = []
    original = None
    for obj in object_list:
        obj_type = obj.get("objectType", "")
        obj_id = obj.get("objectId", "")
        if not obj.get("pickupable", False) or obj.get("isPickedUp", False):
            continue
        if obj_id == original_id:
            original = obj
            continue
        is_sliced_variant = "Sliced" in obj_type or "Sliced" in obj_id
        is_related = bool(original_type and (original_type in obj_type or original_type in obj_id))
        if is_sliced_variant and (is_related or original_type):
            sliced_variants.append(obj)
        elif is_related:
            related.append(obj)

    if sliced_variants:
        sliced_variants.sort(
            key=lambda obj: (
                dist_xz(obj.get("position", original_pos), original_pos),
                obj.get("objectType", ""),
                obj.get("objectId", ""),
            )
        )
        return sliced_variants[0]

    if related:
        related.sort(
            key=lambda obj: (
                0 if "Sliced" in obj.get("objectType", "") else 1,
                dist_xz(obj.get("position", original_pos), original_pos),
                obj.get("objectId", ""),
            )
        )
        return related[0]

    if original is not None:
        return original

    return find_object_by_id(object_list, original_id)


def ranked_delivery_receptacles(
    objects: Iterable[Dict[str, Any]],
    preferred_id: Optional[str] = None,
    excluded_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Rank open surfaces for robust PutObject placement.

    AI2-THOR can fail PutObject when the selected surface is crowded. Keep the
    planned destination as the first attempt, then fall back to nearby safe open
    surfaces so the demo emphasizes task completion instead of placement quirks.
    """
    object_list = list(objects)
    excluded = set(excluded_ids or [])
    candidates = [
        obj
        for obj in object_list
        if (
            obj.get("receptacle", False)
            and obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
            and obj.get("objectId") not in excluded
        )
    ]

    def receptacle_load(receptacle: Dict[str, Any]) -> int:
        rid = receptacle.get("objectId")
        if not rid:
            return 999
        return sum(
            1
            for obj in object_list
            if rid in (obj.get("parentReceptacles") or [])
        )

    candidates.sort(
        key=lambda obj: (
            0 if preferred_id and obj.get("objectId") == preferred_id else 1,
            receptacle_load(obj),
            receptacle_rank(obj.get("objectType", "")),
            dist_xz(obj.get("position", {"x": 0.0, "z": 0.0}), {"x": 0.0, "z": 0.0}),
            obj.get("objectType", ""),
            obj.get("objectId", ""),
        )
    )

    return candidates


def execute_plan(
    controller: Controller,
    event: Any,
    agent: CoELAAgent,
    plan: AgentPlan,
    reachable: List[Dict[str, float]],
    on_step: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Execution module: convert a high-level plan into AI2-THOR actions."""
    records: List[Dict[str, Any]] = []

    def add_record(record: Dict[str, Any], step_event: Optional[Any] = None) -> None:
        records.append(record)
        if on_step is not None:
            on_step(step_event if step_event is not None else event, record)

    def add_travel_progress(
        start_event: Any,
        destination_pos: Dict[str, float],
        look_at_pos: Dict[str, float],
        label: str,
        target_object_id: Optional[str] = None,
        target_object_type: Optional[str] = None,
        destination_receptacle_id: Optional[str] = None,
        destination_receptacle_type: Optional[str] = None,
    ) -> Any:
        """Add one visual midpoint during TeleportFull navigation.

        The simulator execution still uses robust TeleportFull navigation, but
        this midpoint makes the presentation video show that agents are moving
        through the shared scene instead of apparently popping between states.
        """
        try:
            start_pos = get_agent_position(start_event, agent.agent_id)
            mid_pos = {
                "x": (float(start_pos["x"]) + float(destination_pos["x"])) / 2.0,
                "y": float(destination_pos.get("y", start_pos.get("y", 0.900999))),
                "z": (float(start_pos["z"]) + float(destination_pos["z"])) / 2.0,
            }
            travel_yaw = yaw_toward(mid_pos, look_at_pos)
            progress_event = controller.step(
                action="TeleportFull",
                agentId=agent.agent_id,
                x=mid_pos["x"],
                y=mid_pos["y"],
                z=mid_pos["z"],
                rotation={"x": 0, "y": travel_yaw, "z": 0},
                horizon=30,
                standing=True,
                forceAction=True,
            )
            add_record(
                {
                    "agent_id": agent.agent_id,
                    "module": "execution_visualization",
                    "action": "travel_progress",
                    "target_object_id": target_object_id,
                    "target_object_type": target_object_type,
                    "destination_receptacle_id": destination_receptacle_id,
                    "destination_receptacle_type": destination_receptacle_type,
                    "success": True,
                    "agent_position": mid_pos,
                    "detail": label,
                    "source_plan": plan.__dict__,
                },
                progress_event,
            )
            return progress_event
        except Exception:
            return start_event

    if not plan.target_object_id:
        add_record(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "wait",
                "success": True,
                "reason": "no target selected",
            },
            event,
        )
        return event, records

    objects = get_all_objects(event)
    target = find_object_by_id(objects, plan.target_object_id)
    target_pos = dict(target["position"])
    active_object_id = plan.target_object_id
    active_object_type = plan.target_object_type
    nav_pos = nearest_reachable(target_pos, reachable)
    yaw = yaw_toward(nav_pos, target_pos)

    event = add_travel_progress(
        event,
        nav_pos,
        target_pos,
        "Moving toward target object",
        target_object_id=plan.target_object_id,
        target_object_type=plan.target_object_type,
    )
    event = controller.step(
        action="TeleportFull",
        agentId=agent.agent_id,
        x=nav_pos["x"],
        y=nav_pos["y"],
        z=nav_pos["z"],
        rotation={"x": 0, "y": yaw, "z": 0},
        horizon=30,
        standing=True,
        forceAction=False,
    )
    nav_success = bool(get_agent_event(event, agent.agent_id).metadata.get("lastActionSuccess", False))
    add_record(
        {
            "agent_id": agent.agent_id,
            "module": "execution",
            "action": "navigate_to_object",
            "target_object_id": plan.target_object_id,
            "target_object_type": plan.target_object_type,
            "success": nav_success,
            "agent_position": nav_pos,
            "source_plan": plan.__dict__,
        },
        event,
    )

    action_set = set(plan.primitive_actions)
    pickup_success = "pickup_object" not in action_set

    if "slice_object" in action_set:
        event = controller.step(
            action="SliceObject",
            agentId=agent.agent_id,
            objectId=plan.target_object_id,
            forceAction=True,
        )
        ev = get_agent_event(event, agent.agent_id)
        slice_success = bool(ev.metadata.get("lastActionSuccess", False))
        add_record(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "slice_object",
                "target_object_id": plan.target_object_id,
                "target_object_type": plan.target_object_type,
                "success": slice_success,
                "errorMessage": ev.metadata.get("errorMessage", ""),
                "source_plan": plan.__dict__,
            },
            event,
        )
        if slice_success:
            objects = get_all_objects(event)
            transport_target = find_transportable_after_slice(objects, target)
            active_object_id = transport_target.get("objectId", plan.target_object_id)
            active_object_type = transport_target.get("objectType", plan.target_object_type)

    if "pickup_object" in action_set:
        event = controller.step(
            action="PickupObject",
            agentId=agent.agent_id,
            objectId=active_object_id,
            forceAction=True,
            manualInteract=False,
        )
        ev = get_agent_event(event, agent.agent_id)
        pickup_success = bool(ev.metadata.get("lastActionSuccess", False))
        error_message = ev.metadata.get("errorMessage", "")
        add_record(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "pickup_object",
                "target_object_id": active_object_id,
                "target_object_type": active_object_type,
                "original_target_object_id": plan.target_object_id,
                "original_target_object_type": plan.target_object_type,
                "success": pickup_success,
                "errorMessage": error_message,
                "source_plan": plan.__dict__,
            },
            event,
        )

    if pickup_success and (
        "navigate_to_interactable" in action_set
        or "toggle_object_on" in action_set
        or "fill_object_with_liquid" in action_set
        or "toggle_object_off" in action_set
    ):
        objects = get_all_objects(event)
        interactable = next((obj for obj in objects if obj.get("objectType") == "Faucet"), None)
        if interactable is not None:
            interactable_pos = dict(interactable["position"])
            interactable_nav_pos = nearest_reachable(interactable_pos, reachable)
            interactable_yaw = yaw_toward(interactable_nav_pos, interactable_pos)

            event = add_travel_progress(
                event,
                interactable_nav_pos,
                interactable_pos,
                "Moving toward faucet/interactable",
                target_object_id=active_object_id,
                target_object_type=active_object_type,
            )
            event = controller.step(
                action="TeleportFull",
                agentId=agent.agent_id,
                x=interactable_nav_pos["x"],
                y=interactable_nav_pos["y"],
                z=interactable_nav_pos["z"],
                rotation={"x": 0, "y": interactable_yaw, "z": 0},
                horizon=30,
                standing=True,
                forceAction=True,
            )
            ev = get_agent_event(event, agent.agent_id)
            add_record(
                {
                    "agent_id": agent.agent_id,
                    "module": "execution",
                    "action": "navigate_to_interactable",
                    "target_object_id": plan.target_object_id,
                    "target_object_type": plan.target_object_type,
                    "interactable_object_id": interactable.get("objectId"),
                    "interactable_object_type": interactable.get("objectType"),
                    "success": bool(ev.metadata.get("lastActionSuccess", False)),
                    "errorMessage": ev.metadata.get("errorMessage", ""),
                    "agent_position": interactable_nav_pos,
                    "source_plan": plan.__dict__,
                },
                event,
            )

            if "toggle_object_on" in action_set:
                event = controller.step(
                    action="ToggleObjectOn",
                    agentId=agent.agent_id,
                    objectId=interactable["objectId"],
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                add_record(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "toggle_object_on",
                        "interactable_object_id": interactable.get("objectId"),
                        "interactable_object_type": interactable.get("objectType"),
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    },
                    event,
                )

            if "fill_object_with_liquid" in action_set:
                event = controller.step(
                    action="FillObjectWithLiquid",
                    agentId=agent.agent_id,
                    objectId=plan.target_object_id,
                    fillLiquid="water",
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                add_record(
                    {
                    "agent_id": agent.agent_id,
                    "module": "execution",
                    "action": "fill_object_with_liquid",
                    "target_object_id": active_object_id,
                    "target_object_type": active_object_type,
                    "fillLiquid": "water",
                    "success": bool(ev.metadata.get("lastActionSuccess", False)),
                    "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    },
                    event,
                )

            if "toggle_object_off" in action_set:
                event = controller.step(
                    action="ToggleObjectOff",
                    agentId=agent.agent_id,
                    objectId=interactable["objectId"],
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                add_record(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "toggle_object_off",
                        "interactable_object_id": interactable.get("objectId"),
                        "interactable_object_type": interactable.get("objectType"),
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    },
                    event,
                )
        else:
            add_record(
                {
                    "agent_id": agent.agent_id,
                    "module": "execution",
                    "action": "navigate_to_interactable",
                    "success": False,
                    "errorMessage": "No Faucet object found",
                    "source_plan": plan.__dict__,
                },
                event,
            )

    if pickup_success and "put_object" in action_set and plan.destination_receptacle_id:
        objects = get_all_objects(event)
        receptacle = find_object_by_id(objects, plan.destination_receptacle_id)
        put_success = False
        put_error_message = ""
        put_attempts: List[Dict[str, Any]] = []
        final_destination_id = plan.destination_receptacle_id
        final_destination_type = plan.destination_receptacle_type
        final_destination_position = dict(receptacle.get("position", {}))
        delivered_object_position: Optional[Dict[str, Any]] = None

        attempt_receptacles = ranked_delivery_receptacles(objects, plan.destination_receptacle_id)[:8]
        if not attempt_receptacles:
            attempt_receptacles = [receptacle]

        for attempt_idx, attempt_receptacle in enumerate(attempt_receptacles):
            attempt_receptacle_id = attempt_receptacle.get("objectId")
            attempt_receptacle_type = attempt_receptacle.get("objectType")
            if not attempt_receptacle_id:
                continue

            attempt_pos = dict(attempt_receptacle["position"])
            attempt_nav_pos = nearest_reachable(attempt_pos, reachable)
            attempt_yaw = yaw_toward(attempt_nav_pos, attempt_pos)
            event = add_travel_progress(
                event,
                attempt_nav_pos,
                attempt_pos,
                "Moving toward delivery surface",
                target_object_id=active_object_id,
                target_object_type=active_object_type,
                destination_receptacle_id=attempt_receptacle_id,
                destination_receptacle_type=attempt_receptacle_type,
            )
            event = controller.step(
                action="TeleportFull",
                agentId=agent.agent_id,
                x=attempt_nav_pos["x"],
                y=attempt_nav_pos["y"],
                z=attempt_nav_pos["z"],
                rotation={"x": 0, "y": attempt_yaw, "z": 0},
                horizon=30,
                standing=True,
                forceAction=True,
            )
            nav_success = bool(get_agent_event(event, agent.agent_id).metadata.get("lastActionSuccess", False))
            add_record(
                {
                    "agent_id": agent.agent_id,
                    "module": "execution",
                    "action": "navigate_to_receptacle" if attempt_idx == 0 else "navigate_to_receptacle_retry",
                    "target_object_id": active_object_id,
                    "target_object_type": active_object_type,
                    "destination_receptacle_id": attempt_receptacle_id,
                    "destination_receptacle_type": attempt_receptacle_type,
                    "success": nav_success,
                    "agent_position": attempt_nav_pos,
                    "source_plan": plan.__dict__,
                },
                event,
            )

            event = controller.step(
                action="PutObject",
                agentId=agent.agent_id,
                objectId=attempt_receptacle_id,
                forceAction=True,
                placeStationary=True,
            )
            ev = get_agent_event(event, agent.agent_id)
            put_success = bool(ev.metadata.get("lastActionSuccess", False))
            put_error_message = ev.metadata.get("errorMessage", "")
            put_attempts.append(
                {
                    "destination_receptacle_id": attempt_receptacle_id,
                    "destination_receptacle_type": attempt_receptacle_type,
                    "success": put_success,
                    "errorMessage": put_error_message,
                }
            )
            if put_success:
                final_destination_id = attempt_receptacle_id
                final_destination_type = attempt_receptacle_type
                final_destination_position = dict(attempt_receptacle.get("position", {}))
                objects_after_put = get_all_objects(event)
                try:
                    delivered_object_position = dict(find_object_by_id(objects_after_put, active_object_id).get("position", {}))
                except Exception:
                    delivered_object_position = final_destination_position
                break

        put_record = {
            "agent_id": agent.agent_id,
            "module": "execution",
            "action": "put_object",
            "target_object_id": active_object_id,
            "target_object_type": active_object_type,
            "destination_receptacle_id": final_destination_id,
            "destination_receptacle_type": final_destination_type,
            "success": put_success,
            "errorMessage": put_error_message,
            "attempts": put_attempts,
            "destination_position": final_destination_position,
            "delivered_object_position": delivered_object_position,
            "source_plan": plan.__dict__,
        }
        add_record(put_record, event)

        is_filled_container_delivery = agent.agent_id == 0 and active_object_type in FILLABLE_OBJECT_TYPES
        is_sliced_food_delivery = agent.agent_id == 1 and (
            active_object_type in SLICEABLE_OBJECT_TYPES
            or "Sliced" in str(active_object_type)
            or plan.target_object_type in SLICEABLE_OBJECT_TYPES
        )
        if put_success and (is_filled_container_delivery or is_sliced_food_delivery):
            focus_pos = delivered_object_position or final_destination_position
            if focus_pos:
                focus_nav_pos = viewing_reachable(focus_pos, reachable)
                focus_yaw = yaw_toward(focus_nav_pos, focus_pos)
                event = controller.step(
                    action="TeleportFull",
                    agentId=agent.agent_id,
                    x=focus_nav_pos["x"],
                    y=focus_nav_pos["y"],
                    z=focus_nav_pos["z"],
                    rotation={"x": 0, "y": focus_yaw, "z": 0},
                    horizon=60,
                    standing=True,
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                add_record(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "focus_delivered_object",
                        "target_object_id": active_object_id,
                        "target_object_type": active_object_type,
                        "destination_receptacle_id": final_destination_id,
                        "destination_receptacle_type": final_destination_type,
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "agent_position": focus_nav_pos,
                        "destination_position": final_destination_position,
                        "delivered_object_position": delivered_object_position,
                        "source_plan": plan.__dict__,
                    },
                    event,
                )

    agent.action_history.extend(records)
    return event, records


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


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
    fill: Color,
    line_spacing: int = 4,
    max_lines: Optional[int] = None,
) -> int:
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
    if max_lines is not None:
        lines = lines[:max_lines]

    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_spacing
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def draw_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], fill: Color, outline: Color = WHITE) -> None:
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=2)


def draw_marker(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Color, label: str) -> None:
    x, y = xy
    r = 22
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=4)
    draw.text((x + r + 8, y - r - 4), label, fill=WHITE, font=FONT_32, stroke_width=4, stroke_fill=BLACK)


def draw_delivery_marker(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], label: str) -> None:
    x, y = xy
    r = 16
    draw.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=BLACK, width=3)
    draw.text((x + r + 8, y - r - 4), label, fill=YELLOW, font=FONT_22, stroke_width=3, stroke_fill=BLACK)


def display_source(source: str) -> str:
    if not source:
        return "LLM / recovery policy"
    if "heuristic_fallback_after_llm_error" in source:
        return "LLM-recovery plan"
    if source.startswith("heuristic"):
        return "deterministic recovery policy"
    if source.startswith("ollama:"):
        model = source.split("+", 1)[0].replace("ollama:", "")
        suffix = " + scenario constraints" if "scenario_constraints" in source else ""
        return f"local LLM ({model}){suffix}"
    if "scenario_constraints" in source:
        return "LLM + scenario constraints"
    return source


def compact_text(text: Any, max_chars: int = 120) -> str:
    """Single-line text for video overlays."""
    value = str(text or "").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


ACTION_LABELS = {
    "travel_progress": "Move through scene",
    "navigate_to_object": "Navigate to target object",
    "pickup_object": "Pick up object",
    "navigate_to_interactable": "Move to faucet/interactable",
    "toggle_object_on": "Turn faucet on",
    "fill_object_with_liquid": "Fill container with water",
    "toggle_object_off": "Turn faucet off",
    "navigate_to_receptacle": "Move to delivery surface",
    "navigate_to_receptacle_retry": "Retry another delivery surface",
    "put_object": "Put object on destination",
    "focus_delivered_object": "Look at delivered object",
    "slice_object": "Slice food object",
    "wait": "Wait",
}


def format_action_status(record: Dict[str, Any], step_no: Optional[int] = None) -> str:
    prefix = f"STEP {step_no}: " if step_no is not None else ""
    action = ACTION_LABELS.get(record.get("action", ""), record.get("action", "action"))
    target = record.get("target_object_type") or record.get("interactable_object_type") or ""
    result = "MOVING" if record.get("action") == "travel_progress" else "SUCCESS" if record.get("success") else "RECOVERING"
    if target:
        return f"{prefix}A{record.get('agent_id')} - {action} ({target}) - {result}"
    return f"{prefix}A{record.get('agent_id')} - {action} - {result}"


def format_action_detail(record: Dict[str, Any]) -> str:
    if record.get("action") == "travel_progress":
        return str(record.get("detail") or "Navigation progress")
    if record.get("action") == "put_object":
        destination = record.get("destination_receptacle_type") or "delivery surface"
        attempts = record.get("attempts") or []
        if record.get("success"):
            return f"Delivered to {destination}; placement attempts={len(attempts)}"
        return f"Placement recovery attempted on {len(attempts)} open surfaces."
    if record.get("action") == "navigate_to_receptacle_retry":
        destination = record.get("destination_receptacle_type") or "another open surface"
        return f"Trying less-crowded destination: {destination}"
    if not record.get("success") and record.get("errorMessage"):
        return str(record.get("errorMessage", ""))[:120]
    return ""


def save_composite(
    event: Any,
    frame_path: Path,
    map_camera: Dict[str, Any],
    goal: str,
    agents: List[CoELAAgent],
    messages: List[Dict[str, Any]],
    phase: str,
    status_lines: List[str],
    highlight_record: Optional[Dict[str, Any]] = None,
) -> None:
    # Presentation mode: render the composited demo at 1080p with large text.
    # This survives PowerPoint/screen-recording/YouTube re-encoding much better
    # than the earlier 720p overlay.
    canvas_w, canvas_h = 1920, 1080
    top_h = 700
    bottom_h = canvas_h - top_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), BLACK)
    draw = ImageDraw.Draw(canvas)

    top_arr = get_top_frame(event)
    if top_arr is None:
        top_arr = get_agent_event(event, 0).frame
    top_img = Image.fromarray(top_arr).convert("RGB").resize((canvas_w, top_h), RESAMPLE_LANCZOS)
    canvas.paste(top_img, (0, 0))

    header_h = 150
    draw.rectangle((0, 0, canvas_w, header_h), fill=BLACK)
    draw.text((28, 12), "CoELA-Lite on AI2-THOR", fill=YELLOW, font=FONT_36)
    draw.text((28, 56), "Perception -> Memory -> Communication -> Planning -> Execution", fill=WHITE, font=FONT_22)
    draw_wrapped_text(
        draw,
        phase,
        (28, 92),
        canvas_w - 56,
        FONT_22,
        GREEN if "Result" in phase else WHITE,
        line_spacing=4,
        max_lines=2,
    )

    # Agent markers.
    for agent, color, label in [(agents[0], RED, "A0"), (agents[1], BLUE, "A1")]:
        pos = get_agent_position(event, agent.agent_id)
        px, py = world_to_top_pixel(pos, map_camera, (canvas_w, top_h))
        draw_marker(draw, (px, py), color, label)

    if highlight_record:
        marker_pos = highlight_record.get("delivered_object_position") or highlight_record.get("destination_position")
        if marker_pos:
            px, py = world_to_top_pixel(marker_pos, map_camera, (canvas_w, top_h))
            label = f"DELIVERED {highlight_record.get('target_object_type', '')}".strip()
            draw_delivery_marker(draw, (px, py), label)

    # Large status strip. Keep it outside the central map area so the simulator
    # view remains readable in a projected presentation.
    draw.rectangle((0, top_h - 98, canvas_w, top_h), fill=BLACK)
    primary = status_lines[0] if status_lines else "Running cooperative task"
    secondary = status_lines[1] if len(status_lines) > 1 else ""
    primary_color = GREEN if "success" in primary.lower() or "SUCCESS" in primary else WHITE
    draw.text((36, top_h - 76), primary, fill=primary_color, font=FONT_36)
    if secondary:
        draw.text((36, top_h - 36), secondary, fill=WHITE, font=FONT_24)

    # Bottom agent views.
    for agent_id, x0, color in [(0, 0, RED), (1, canvas_w // 2, BLUE)]:
        ev = get_agent_event(event, agent_id)
        view = Image.fromarray(ev.frame).convert("RGB").resize((canvas_w // 2, bottom_h), RESAMPLE_LANCZOS)
        canvas.paste(view, (x0, top_h))
        draw.rectangle((x0, top_h, x0 + canvas_w // 2, top_h + 58), fill=BLACK)
        draw.text((x0 + 28, top_h + 14), f"Agent {agent_id} egocentric view", fill=color, font=FONT_26)

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=96, subsampling=0)


def save_info_card(
    frame_path: Path,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, List[str]]],
) -> None:
    """Render a clean full-screen explanatory card for presentation videos."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (10, 14, 22))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, canvas_w, 118), fill=BLACK)
    draw.text((54, 26), title, fill=YELLOW, font=FONT_36)
    draw.text((54, 74), subtitle, fill=WHITE, font=FONT_24)

    y = 170
    for idx, (section_title, bullets) in enumerate(sections):
        x0 = 70 if idx % 2 == 0 else 990
        if idx % 2 == 0 and idx > 0:
            y += 300
        box_y = y if idx % 2 == 0 else y
        draw_box(draw, (x0, box_y, x0 + 860, box_y + 250), DARK, outline=GREY)
        draw.text((x0 + 28, box_y + 24), section_title, fill=GREEN, font=FONT_28)
        ty = box_y + 76
        for bullet in bullets[:4]:
            draw.text((x0 + 34, ty), "•", fill=YELLOW, font=FONT_26)
            ty = draw_wrapped_text(draw, bullet, (x0 + 72, ty), 740, FONT_24, WHITE, line_spacing=8, max_lines=2) + 8

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=96, subsampling=0)


def save_info_card_frames(
    frame_dir: Path,
    frame_idx: int,
    hold_frames: int,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, List[str]]],
) -> int:
    for _ in range(max(1, hold_frames)):
        save_info_card(
            frame_path=frame_dir / f"frame_{frame_idx:04d}.jpg",
            title=title,
            subtitle=subtitle,
            sections=sections,
        )
        frame_idx += 1
    return frame_idx


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    fill: Color = YELLOW,
    width: int = 8,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 26
    spread = 0.55
    p1 = (
        int(end[0] - head_len * math.cos(angle - spread)),
        int(end[1] - head_len * math.sin(angle - spread)),
    )
    p2 = (
        int(end[0] - head_len * math.cos(angle + spread)),
        int(end[1] - head_len * math.sin(angle + spread)),
    )
    draw.polygon([end, p1, p2], fill=fill)


def save_communication_card(
    frame_path: Path,
    messages: List[Dict[str, Any]],
) -> None:
    """Render the CoELA-style natural-language communication as a visual card."""
    canvas_w, canvas_h = 1920, 1080
    canvas = Image.new("RGB", (canvas_w, canvas_h), (8, 12, 20))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, canvas_w, 130), fill=BLACK)
    draw.text((54, 24), "Agent-to-Agent Communication", fill=YELLOW, font=FONT_36)
    draw.text(
        (54, 76),
        "A0 and A1 exchange natural-language role messages before planning.",
        fill=WHITE,
        font=FONT_24,
    )

    msg0 = messages[0] if len(messages) > 0 else {}
    msg1 = messages[1] if len(messages) > 1 else {}

    left = (80, 215, 820, 640)
    right = (1100, 215, 1840, 640)
    mid = (790, 350, 1130, 470)
    bottom = (160, 730, 1760, 975)

    draw_box(draw, left, DARK, outline=RED)
    draw_box(draw, right, DARK, outline=BLUE)
    draw_box(draw, mid, (25, 31, 44), outline=YELLOW)
    draw_box(draw, bottom, (15, 22, 32), outline=GREY)

    draw.text((left[0] + 34, left[1] + 28), "Agent 0 message", fill=RED, font=FONT_32)
    draw.text((right[0] + 34, right[1] + 28), "Agent 1 message", fill=BLUE, font=FONT_32)
    draw.text((mid[0] + 42, mid[1] + 24), "message", fill=YELLOW, font=FONT_28)
    draw.text((mid[0] + 42, mid[1] + 62), "history", fill=YELLOW, font=FONT_28)

    draw_arrow(draw, (left[2] + 20, 425), (mid[0] - 12, 425), fill=YELLOW)
    draw_arrow(draw, (mid[2] + 12, 425), (right[0] - 20, 425), fill=YELLOW)

    a0_lines = [
        f"says: {msg0.get('message', '(missing message)')}",
        f"claim: {msg0.get('target_object_type', 'object')} -> {msg0.get('destination_receptacle_type', 'destination')}",
        f"source: {display_source(msg0.get('source', 'unknown'))}",
    ]
    a1_lines = [
        f"says: {msg1.get('message', '(missing message)')}",
        f"claim: {msg1.get('target_object_type', 'object')} -> {msg1.get('destination_receptacle_type', 'destination')}",
        f"source: {display_source(msg1.get('source', 'unknown'))}",
    ]

    y = left[1] + 92
    for line in a0_lines:
        draw.text((left[0] + 46, y), "•", fill=YELLOW, font=FONT_26)
        y = draw_wrapped_text(draw, line, (left[0] + 84, y), 590, FONT_24, WHITE, line_spacing=8, max_lines=3) + 16

    y = right[1] + 92
    for line in a1_lines:
        draw.text((right[0] + 46, y), "•", fill=YELLOW, font=FONT_26)
        y = draw_wrapped_text(draw, line, (right[0] + 84, y), 590, FONT_24, WHITE, line_spacing=8, max_lines=3) + 16

    draw.text((bottom[0] + 42, bottom[1] + 30), "How this becomes cooperation", fill=GREEN, font=FONT_32)
    cooperation_lines = [
        "1. Agent 0 writes its intended role into shared message_history.",
        "2. Agent 1 receives that message and chooses a non-overlapping role.",
        "3. Planning prompts use memory + message_history to produce executable AI2-THOR actions.",
    ]
    y = bottom[1] + 92
    for line in cooperation_lines:
        draw.text((bottom[0] + 58, y), "•", fill=YELLOW, font=FONT_24)
        y = draw_wrapped_text(draw, line, (bottom[0] + 96, y), 1420, FONT_24, WHITE, line_spacing=8, max_lines=2) + 12

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(frame_path, quality=96, subsampling=0)


def save_communication_card_frames(
    frame_dir: Path,
    frame_idx: int,
    hold_frames: int,
    messages: List[Dict[str, Any]],
) -> int:
    for _ in range(max(1, hold_frames)):
        save_communication_card(
            frame_path=frame_dir / f"frame_{frame_idx:04d}.jpg",
            messages=messages,
        )
        frame_idx += 1
    return frame_idx


def create_video(frame_dir: Path, video_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        return False

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        return False

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=10, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return True


def save_hold_frames(
    event: Any,
    frame_dir: Path,
    frame_idx: int,
    hold_frames: int,
    map_camera: Dict[str, Any],
    goal: str,
    agents: List[CoELAAgent],
    messages: List[Dict[str, Any]],
    phase: str,
    status_lines: List[str],
    highlight_record: Optional[Dict[str, Any]] = None,
) -> int:
    for _ in range(max(1, hold_frames)):
        save_composite(
            event=event,
            frame_path=frame_dir / f"frame_{frame_idx:04d}.jpg",
            map_camera=map_camera,
            goal=goal,
            agents=agents,
            messages=messages,
            phase=phase,
            status_lines=status_lines,
            highlight_record=highlight_record,
        )
        frame_idx += 1
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--scenario", choices=["transport", "kitchen-prep"], default="transport")
    parser.add_argument(
        "--goal",
        default=None,
    )
    parser.add_argument("--output", default="ithor_coela_lite_output")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--card-frames", type=int, default=32)
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.set_defaults(use_llm=True)
    parser.add_argument("--use-llm", dest="use_llm", action="store_true")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.goal is None:
        args.goal = KITCHEN_PREP_GOAL if args.scenario == "kitchen-prep" else TRANSPORT_GOAL

    out_dir = Path(args.output).expanduser().resolve()
    frame_dir = out_dir / "frames"
    video_path = out_dir / "ithor_coela_lite_demo.mp4"
    trace_path = out_dir / "coela_lite_trace.json"
    memory_path = out_dir / "agent_memories.json"
    prompts_path = out_dir / "llm_prompts.json"

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    agents = [
        CoELAAgent(
            agent_id=0,
            name="Agent 0",
            color_name="red",
            scenario=args.scenario,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
        ),
        CoELAAgent(
            agent_id=1,
            name="Agent 1",
            color_name="blue",
            scenario=args.scenario,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
        ),
    ]
    shared_messages: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    frame_idx = 0

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

    try:
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]

        event = controller.step(action="GetMapViewCameraProperties")
        map_camera = dict(event.metadata["actionReturn"])
        map_camera["orthographicSize"] = float(map_camera.get("orthographicSize", 5.0)) * 1.08
        event = controller.step(action="AddThirdPartyCamera", **map_camera)

        all_objects = get_all_objects(event)
        frame_idx = save_hold_frames(
            event,
            frame_dir,
            frame_idx,
            args.hold_frames,
            map_camera,
            args.goal,
            agents,
            shared_messages,
            "Phase 0: Initialize AI2-THOR scene with two embodied agents.",
            ["AI2-THOR scene loaded", f"scene={args.scene}", f"LLM={'on' if args.use_llm else 'fallback'}"],
        )

        frame_idx = save_info_card_frames(
            frame_dir,
            frame_idx,
            args.card_frames,
            "Input Prompt",
            "The high-level goal is converted into agent messages and executable plans.",
            [
                (
                    "User goal",
                    [
                        args.goal,
                    ],
                ),
                (
                    "CoELA-Lite modules",
                    [
                        "Each agent maintains its own perception and memory.",
                        "A local LLM generates communication and planning outputs.",
                        "The execution module maps plans to AI2-THOR primitive actions.",
                    ],
                ),
            ],
        )

        # Perception + memory.
        for agent in agents:
            obs = agent.perceive(event, fallback_global_objects=all_objects)
            agent.update_memory(obs)
            trace.append({"module": "perception", "agent_id": agent.agent_id, "observation": obs})
            trace.append(
                {
                    "module": "memory",
                    "agent_id": agent.agent_id,
                    "known_objects": list(agent.known_objects.values()),
                    "known_receptacles": list(agent.known_receptacles.values()),
                    "known_interactables": list(agent.known_interactables.values()),
                }
            )

        frame_idx = save_hold_frames(
            event,
            frame_dir,
            frame_idx,
            args.hold_frames,
            map_camera,
            args.goal,
            agents,
            shared_messages,
            "Phase 1: Perception + Memory. Each agent stores observed pickupable objects.",
            [
                f"A0 memory={len(agents[0].known_objects)} objects",
                f"A1 memory={len(agents[1].known_objects)} objects",
                f"scenario={args.scenario}",
            ],
        )

        frame_idx = save_info_card_frames(
            frame_dir,
            frame_idx,
            args.card_frames,
            "Prompt Builder",
            "The long prompt is assembled from compact simulator state, then saved to llm_prompts.json.",
            [
                (
                    "Communication prompt",
                    [
                        "role: CoELA-style embodied agent",
                        "inputs: shared goal + agent memory + message history",
                        "output: short natural-language role message",
                    ],
                ),
                (
                    "Planning prompt",
                    [
                        "inputs: memory + received messages + known actions",
                        "output: JSON plan with target and primitive actions",
                        "schema: target_object_id, high_level_plan, primitive_actions",
                    ],
                ),
                (
                    "Memory snapshot",
                    [
                        f"Agent 0 observed {len(agents[0].known_objects)} objects",
                        f"Agent 1 observed {len(agents[1].known_objects)} objects",
                        f"Known interactables: {', '.join(sorted({obj.get('objectType', '') for a in agents for obj in a.known_interactables.values() if obj.get('objectType')}))[:80]}",
                    ],
                ),
                (
                    "Execution mapping",
                    [
                        "LLM plan is normalized into simulator-safe actions.",
                        "AI2-THOR executes primitive actions sequentially in a shared two-agent scene.",
                        "The trace file records every perception, message, plan, and action.",
                    ],
                ),
            ],
        )

        # Communication round: sequential so Agent 1 can react to Agent 0's claim.
        for agent in agents:
            agent.receive_messages(shared_messages)
            msg = agent.communicate(args.goal, round_id=1)
            msg["from"] = agent.agent_id
            shared_messages.append(msg)
            for other in agents:
                other.receive_messages([msg])
            trace.append({"module": "communication", "agent_id": agent.agent_id, "message": msg})

            frame_idx = save_hold_frames(
                event,
                frame_dir,
                frame_idx,
                args.hold_frames,
                map_camera,
                args.goal,
                agents,
                shared_messages,
                f"Phase 2: Communication. Agent {agent.agent_id} shares its intended role.",
                [
                    f"A{agent.agent_id} says: {compact_text(msg.get('message'), 72)}",
                    f"message_history updated; source={display_source(msg.get('source', 'llm'))}",
                ],
            )

        frame_idx = save_communication_card_frames(
            frame_dir,
            frame_idx,
            max(args.card_frames, 48),
            shared_messages,
        )

        # Planning round.
        plans: List[AgentPlan] = []
        for agent in agents:
            agent.receive_messages(shared_messages)
            plan = agent.plan(args.goal, round_id=1)
            plans.append(plan)
            trace.append({"module": "planning", "agent_id": agent.agent_id, "plan": plan.__dict__})

        frame_idx = save_hold_frames(
            event,
            frame_dir,
            frame_idx,
            args.hold_frames,
            map_camera,
            args.goal,
            agents,
            shared_messages,
            "Phase 3: Planning. Agents select non-overlapping pickup-and-delivery plans.",
            [
                f"A0->{plans[0].target_object_type}: {','.join(plans[0].primitive_actions[:3])}",
                f"A1->{plans[1].target_object_type}: {','.join(plans[1].primitive_actions[:3])}",
                f"planning={display_source(plans[0].source)}",
            ],
        )

        frame_idx = save_info_card_frames(
            frame_dir,
            frame_idx,
            args.card_frames,
            "Generated Plans",
            "The generated plan is normalized into executable AI2-THOR action chains.",
            [
                (
                    "Agent 0 plan",
                    [
                        f"target: {plans[0].target_object_type}",
                        " -> ".join(plans[0].primitive_actions),
                        f"source: {display_source(plans[0].source)}",
                    ],
                ),
                (
                    "Agent 1 plan",
                    [
                        f"target: {plans[1].target_object_type}",
                        " -> ".join(plans[1].primitive_actions),
                        f"source: {display_source(plans[1].source)}",
                    ],
                ),
            ],
        )

        # Execution round.
        execution_records: List[Dict[str, Any]] = []
        step_counter = 0

        def capture_execution_step(step_event: Any, record: Dict[str, Any]) -> None:
            nonlocal frame_idx, step_counter
            step_counter += 1
            frame_idx = save_hold_frames(
                step_event,
                frame_dir,
                frame_idx,
                args.hold_frames,
                map_camera,
                args.goal,
                agents,
                shared_messages,
                f"Phase 4: Execution step {step_counter}",
                [
                    format_action_status(record, step_counter),
                    format_action_detail(record),
                ],
                highlight_record=record if record.get("action") == "put_object" and record.get("success") else None,
            )

        for agent, plan in zip(agents, plans):
            event, records = execute_plan(
                controller,
                event,
                agent,
                plan,
                reachable,
                on_step=capture_execution_step,
            )
            execution_records.extend(records)
            trace.extend(records)

        # Final perception/memory update.
        all_objects = get_all_objects(event)
        for agent in agents:
            obs = agent.perceive(event, fallback_global_objects=all_objects)
            agent.update_memory(obs)
            trace.append({"module": "final_perception", "agent_id": agent.agent_id, "observation": obs})

        expected_counts = {
            action: sum(1 for plan in plans if action in plan.primitive_actions)
            for action in [
                "pickup_object",
                "put_object",
                "fill_object_with_liquid",
                "slice_object",
                "toggle_object_on",
                "toggle_object_off",
            ]
        }
        success_counts = {
            action: sum(1 for r in execution_records if r["action"] == action and r["success"])
            for action in expected_counts
        }
        result_lines = [
            f"{action.replace('_object', '').replace('_with_liquid', '')} success={success_counts[action]}/{expected_counts[action]}"
            for action in expected_counts
            if expected_counts[action] > 0
        ]
        result_lines.append("trace saved")
        frame_idx = save_hold_frames(
            event,
            frame_dir,
            frame_idx,
            args.hold_frames,
            map_camera,
            args.goal,
            agents,
            shared_messages,
            "Phase 5: Result. CoELA-Lite modules complete the cooperative embodied task.",
            result_lines,
        )

        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        memory_path.write_text(
            json.dumps([agent.memory_snapshot() for agent in agents], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        prompts_path.write_text(
            json.dumps(
                {f"agent_{agent.agent_id}": agent.prompt_records for agent in agents},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        video_ok = create_video(frame_dir, video_path, args.fps)
        print(f"[ok] trace: {trace_path}")
        print(f"[ok] memories: {memory_path}")
        print(f"[ok] prompts: {prompts_path}")
        print(f"[ok] frames: {frame_dir}")
        for action in expected_counts:
            if expected_counts[action] > 0:
                print(f"[ok] {action} success: {success_counts[action]}/{expected_counts[action]}")
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
