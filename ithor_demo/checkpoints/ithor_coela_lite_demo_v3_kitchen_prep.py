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

FILLABLE_OBJECT_TYPES = ["Mug", "Cup", "Bowl"]
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
    "Coordinate a longer kitchen-preparation task. Agent 0 should fill a mug or cup with "
    "water using the faucet and deliver it to a countertop. Agent 1 should slice a food "
    "object such as lettuce, tomato, potato, bread, or apple."
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


def call_ollama_json(model: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a CoELA-style embodied-agent module. "
                    "Return valid JSON only. No markdown, no prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    return extract_json(content)


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
            if obj.get("objectType") in SAFE_DELIVERY_RECEPTACLE_TYPES
        ]
        candidates.sort(key=lambda o: (receptacle_rank(o["objectType"]), o["objectType"], o["objectId"] or ""))
        return candidates

    def _preferred_delivery_receptacle(self) -> Optional[Dict[str, Any]]:
        candidates = self._candidate_receptacles()
        return candidates[0] if candidates else None

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
        receptacle = self._preferred_delivery_receptacle()
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
            payload["message"] = (
                f"{self.name}: I will slice the {payload.get('target_object_type', 'lettuce')} "
                "as the food-preparation subtask while you handle the water."
            )

        payload["scenario_constraints_applied"] = True
        return payload

    def _apply_kitchen_prep_plan_constraints(self, plan: AgentPlan) -> AgentPlan:
        if self.scenario != "kitchen-prep":
            return plan

        target = self._kitchen_prep_target()
        receptacle = self._preferred_delivery_receptacle()
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
            plan.destination_receptacle_id = None
            plan.destination_receptacle_type = None
            plan.primitive_actions = ["navigate_to_object", "slice_object"]
            plan.high_level_plan = f"Navigate to the {plan.target_object_type} and slice it."

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
                "- Agent 1 should claim a sliceable food object such as Lettuce/Tomato/Potato/Bread/Apple and slice it.\n"
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
                    raise ValueError("LLM communication JSON has no message field")
                msg = self._ensure_safe_delivery_receptacle_fields(msg)
                msg = self._apply_kitchen_prep_message_constraints(msg)
                return msg
            except Exception as exc:
                fallback = self.heuristic_communication(goal, round_id)
                fallback["llm_error"] = repr(exc)
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
                "- Agent 1 plan should include: navigate_to_object, slice_object.\n"
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

    def heuristic_plan(self, goal: str) -> AgentPlan:
        candidates = self._candidate_objects()
        receptacle = self._preferred_delivery_receptacle()
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
        if self.use_llm:
            try:
                raw = call_ollama_json(self.llm_model, prompt)
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
                    source=f"ollama:{self.llm_model}",
                )
                if plan.target_object_id and plan.target_object_id not in self.known_objects:
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
                plan = self._apply_kitchen_prep_plan_constraints(plan)
                self.current_plan = plan
                return plan
            except Exception as exc:
                plan = self.heuristic_plan(goal)
                plan.source = f"heuristic_fallback_after_llm_error:{repr(exc)}"
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


def execute_plan(
    controller: Controller,
    event: Any,
    agent: CoELAAgent,
    plan: AgentPlan,
    reachable: List[Dict[str, float]],
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Execution module: convert a high-level plan into AI2-THOR actions."""
    records: List[Dict[str, Any]] = []
    if not plan.target_object_id:
        records.append(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "wait",
                "success": True,
                "reason": "no target selected",
            }
        )
        return event, records

    objects = get_all_objects(event)
    target = find_object_by_id(objects, plan.target_object_id)
    target_pos = dict(target["position"])
    nav_pos = nearest_reachable(target_pos, reachable)
    yaw = yaw_toward(nav_pos, target_pos)

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
    records.append(
        {
            "agent_id": agent.agent_id,
            "module": "execution",
            "action": "navigate_to_object",
            "target_object_id": plan.target_object_id,
            "target_object_type": plan.target_object_type,
            "success": nav_success,
            "agent_position": nav_pos,
            "source_plan": plan.__dict__,
        }
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
        records.append(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "slice_object",
                "target_object_id": plan.target_object_id,
                "target_object_type": plan.target_object_type,
                "success": slice_success,
                "errorMessage": ev.metadata.get("errorMessage", ""),
                "source_plan": plan.__dict__,
            }
        )

    if "pickup_object" in action_set:
        event = controller.step(
            action="PickupObject",
            agentId=agent.agent_id,
            objectId=plan.target_object_id,
            forceAction=True,
            manualInteract=False,
        )
        ev = get_agent_event(event, agent.agent_id)
        pickup_success = bool(ev.metadata.get("lastActionSuccess", False))
        error_message = ev.metadata.get("errorMessage", "")
        records.append(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "pickup_object",
                "target_object_id": plan.target_object_id,
                "target_object_type": plan.target_object_type,
                "success": pickup_success,
                "errorMessage": error_message,
                "source_plan": plan.__dict__,
            }
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
            records.append(
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
                }
            )

            if "toggle_object_on" in action_set:
                event = controller.step(
                    action="ToggleObjectOn",
                    agentId=agent.agent_id,
                    objectId=interactable["objectId"],
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                records.append(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "toggle_object_on",
                        "interactable_object_id": interactable.get("objectId"),
                        "interactable_object_type": interactable.get("objectType"),
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    }
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
                records.append(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "fill_object_with_liquid",
                        "target_object_id": plan.target_object_id,
                        "target_object_type": plan.target_object_type,
                        "fillLiquid": "water",
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    }
                )

            if "toggle_object_off" in action_set:
                event = controller.step(
                    action="ToggleObjectOff",
                    agentId=agent.agent_id,
                    objectId=interactable["objectId"],
                    forceAction=True,
                )
                ev = get_agent_event(event, agent.agent_id)
                records.append(
                    {
                        "agent_id": agent.agent_id,
                        "module": "execution",
                        "action": "toggle_object_off",
                        "interactable_object_id": interactable.get("objectId"),
                        "interactable_object_type": interactable.get("objectType"),
                        "success": bool(ev.metadata.get("lastActionSuccess", False)),
                        "errorMessage": ev.metadata.get("errorMessage", ""),
                        "source_plan": plan.__dict__,
                    }
                )
        else:
            records.append(
                {
                    "agent_id": agent.agent_id,
                    "module": "execution",
                    "action": "navigate_to_interactable",
                    "success": False,
                    "errorMessage": "No Faucet object found",
                    "source_plan": plan.__dict__,
                }
            )

    if pickup_success and "put_object" in action_set and plan.destination_receptacle_id:
        objects = get_all_objects(event)
        receptacle = find_object_by_id(objects, plan.destination_receptacle_id)
        receptacle_pos = dict(receptacle["position"])
        receptacle_nav_pos = nearest_reachable(receptacle_pos, reachable)
        receptacle_yaw = yaw_toward(receptacle_nav_pos, receptacle_pos)

        event = controller.step(
            action="TeleportFull",
            agentId=agent.agent_id,
            x=receptacle_nav_pos["x"],
            y=receptacle_nav_pos["y"],
            z=receptacle_nav_pos["z"],
            rotation={"x": 0, "y": receptacle_yaw, "z": 0},
            horizon=30,
            standing=True,
            forceAction=False,
        )
        nav_receptacle_success = bool(get_agent_event(event, agent.agent_id).metadata.get("lastActionSuccess", False))
        records.append(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "navigate_to_receptacle",
                "target_object_id": plan.target_object_id,
                "target_object_type": plan.target_object_type,
                "destination_receptacle_id": plan.destination_receptacle_id,
                "destination_receptacle_type": plan.destination_receptacle_type,
                "success": nav_receptacle_success,
                "agent_position": receptacle_nav_pos,
                "source_plan": plan.__dict__,
            }
        )

        event = controller.step(
            action="PutObject",
            agentId=agent.agent_id,
            objectId=plan.destination_receptacle_id,
            forceAction=True,
            placeStationary=True,
        )
        ev = get_agent_event(event, agent.agent_id)
        put_success = bool(ev.metadata.get("lastActionSuccess", False))
        put_error_message = ev.metadata.get("errorMessage", "")
        records.append(
            {
                "agent_id": agent.agent_id,
                "module": "execution",
                "action": "put_object",
                "target_object_id": plan.target_object_id,
                "target_object_type": plan.target_object_type,
                "destination_receptacle_id": plan.destination_receptacle_id,
                "destination_receptacle_type": plan.destination_receptacle_type,
                "success": put_success,
                "errorMessage": put_error_message,
                "source_plan": plan.__dict__,
            }
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
    r = 15
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=4)
    draw.text((x + r + 6, y - r - 3), label, fill=WHITE, font=FONT_22, stroke_width=3, stroke_fill=BLACK)


def save_composite(
    event: Any,
    frame_path: Path,
    map_camera: Dict[str, Any],
    goal: str,
    agents: List[CoELAAgent],
    messages: List[Dict[str, Any]],
    phase: str,
    status_lines: List[str],
) -> None:
    canvas_w, canvas_h = 1280, 720
    top_h = 470
    bottom_h = canvas_h - top_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), BLACK)
    draw = ImageDraw.Draw(canvas)

    top_arr = get_top_frame(event)
    if top_arr is None:
        top_arr = get_agent_event(event, 0).frame
    top_img = Image.fromarray(top_arr).convert("RGB").resize((canvas_w, top_h), RESAMPLE_LANCZOS)
    canvas.paste(top_img, (0, 0))

    draw.rectangle((0, 0, canvas_w, 76), fill=BLACK)
    draw.text((18, 10), "CoELA-Lite on AI2-THOR: Perception -> Memory -> Communication -> Planning -> Execution", fill=YELLOW, font=FONT_22)
    draw.text((18, 42), phase, fill=WHITE, font=FONT_18)

    # Goal panel.
    draw_box(draw, (18, 88, 500, 182), DARK)
    draw.text((36, 102), "Shared goal", fill=YELLOW, font=FONT_18)
    draw_wrapped_text(draw, goal, (36, 130), 430, FONT_16, WHITE, max_lines=2)

    # Module panel.
    draw_box(draw, (760, 88, 1260, 290), DARK)
    draw.text((780, 104), "Agent modules", fill=YELLOW, font=FONT_18)
    modules = ["Perception", "Memory", "Communication", "Planning", "Execution"]
    x = 780
    y = 138
    for i, module in enumerate(modules):
        draw.text((x, y), module, fill=GREEN if module in phase else WHITE, font=FONT_16)
        y += 27

    # Messages / plans panel.
    draw_box(draw, (18, 196, 690, 402), DARK)
    draw.text((36, 212), "Communication / Plans", fill=YELLOW, font=FONT_18)
    y = 240
    for msg in messages[-4:]:
        color = RED if int(msg.get("agent_id", msg.get("from", 0))) == 0 else BLUE
        text = msg.get("message", "")
        draw.text((36, y), f"A{msg.get('agent_id', msg.get('from'))}:", fill=color, font=FONT_16)
        y = draw_wrapped_text(draw, text, (80, y), 570, FONT_14, WHITE, max_lines=2) + 4
    for agent in agents:
        if agent.current_plan:
            color = RED if agent.agent_id == 0 else BLUE
            plan = agent.current_plan
            line = f"A{agent.agent_id} plan: {plan.target_object_type} / {plan.high_level_plan}"
            draw_wrapped_text(draw, line, (36, y), 620, FONT_14, color, max_lines=1)
            y += 22

    # Agent markers.
    for agent, color, label in [(agents[0], RED, "A0"), (agents[1], BLUE, "A1")]:
        pos = get_agent_position(event, agent.agent_id)
        px, py = world_to_top_pixel(pos, map_camera, (canvas_w, top_h))
        draw_marker(draw, (px, py), color, label)

    # Status strip.
    draw.rectangle((0, top_h - 42, canvas_w, top_h), fill=BLACK)
    x = 18
    for line in status_lines[:4]:
        draw.text((x, top_h - 30), line, fill=GREEN if "success" in line.lower() else WHITE, font=FONT_16)
        x += 310

    # Bottom agent views.
    for agent_id, x0, color in [(0, 0, RED), (1, canvas_w // 2, BLUE)]:
        ev = get_agent_event(event, agent_id)
        view = Image.fromarray(ev.frame).convert("RGB").resize((canvas_w // 2, bottom_h), RESAMPLE_LANCZOS)
        canvas.paste(view, (x0, top_h))
        draw.rectangle((x0, top_h, x0 + canvas_w // 2, top_h + 42), fill=BLACK)
        draw.text((x0 + 18, top_h + 10), f"Agent {agent_id} egocentric view", fill=color, font=FONT_18)

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
                [f"A{agent.agent_id} message generated", f"source={msg.get('source', 'llm')}"],
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
                f"plan_source={plans[0].source}",
            ],
        )

        # Execution round.
        execution_records: List[Dict[str, Any]] = []
        for agent, plan in zip(agents, plans):
            event, records = execute_plan(controller, event, agent, plan, reachable)
            execution_records.extend(records)
            trace.extend(records)
            status = [
                f"A{r['agent_id']} {r['action']} {'success' if r['success'] else 'failed'}"
                for r in records
            ]
            frame_idx = save_hold_frames(
                event,
                frame_dir,
                frame_idx,
                args.hold_frames,
                map_camera,
                args.goal,
                agents,
                shared_messages,
                f"Phase 4: Execution. Agent {agent.agent_id} executes its high-level plan.",
                status,
            )

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
