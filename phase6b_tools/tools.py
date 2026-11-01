"""Phase 6b: the tools, and the checks that stand between the model and them.

The model can only write text. A line of that text may PROPOSE an action. This file decides
whether the proposal is well-formed and permitted, and only then runs it. The tools work on a
small in-memory dictionary: nothing here touches files, the network or real data.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_CALL_CHARS = 200
MAX_RESULT_CHARS = 100
MAX_NUMBER = 1000
SUBJECT = re.compile(r"[a-z]{1,20}( [a-z]{1,20})?")


class CallRejected(Exception):
    """A proposal was refused. `public` goes back to the model; `detail` goes in the log only."""

    def __init__(self, public, detail):
        super().__init__(detail)
        self.public, self.detail = public, detail


def load_facts():
    """Subject -> fact, from data/instructions.txt ("About the owl." / "The owl was wise and old.")."""
    blocks = [b.split("\n") for b in (DATA / "instructions.txt").read_text(encoding="utf-8").strip().split("\n\n")]
    return {prompt.removeprefix("About the ").rstrip("."): fact for prompt, fact in blocks}


# ---------------------------------------------------------------------------
# Part 1: what each tool accepts. Anything not described here is refused.
# ---------------------------------------------------------------------------

def is_number(value):
    # bool is excluded because Python counts True as the number 1.
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_NUMBER


def is_subject(value):
    return isinstance(value, str) and SUBJECT.fullmatch(value) is not None


@dataclass(frozen=True)
class Tool:
    fields: dict                  # field name -> function that checks its value
    needs_approval: bool = False  # True: a person must also confirm each individual call


TOOLS = {
    "add": Tool({"a": is_number, "b": is_number}),
    "lookup": Tool({"subject": is_subject}),
    "delete": Tool({"subject": is_subject}, needs_approval=True),
}
READ_ONLY = frozenset({"add", "lookup"})


# ---------------------------------------------------------------------------
# Part 2: validating a proposal. The order matters: shape first, then permission.
# ---------------------------------------------------------------------------

def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")   # otherwise {"tool":"add","tool":"delete"} silently means delete
    return dict(pairs)


def parse_call(line):
    """Text -> (tool name, arguments), or raise CallRejected."""
    if len(line) > MAX_CALL_CHARS:
        raise CallRejected("invalid call", f"call is {len(line)} characters, limit {MAX_CALL_CHARS}")
    try:
        call = json.loads(line, object_pairs_hook=_no_duplicate_keys)
    except ValueError as error:
        raise CallRejected("invalid call", f"not valid JSON: {error}")
    if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
        raise CallRejected("invalid call", "no tool name")
    name = call.pop("tool")
    if name not in TOOLS:
        raise CallRejected("unknown tool", f"unknown tool {name!r}")
    fields = TOOLS[name].fields
    if set(call) != set(fields):
        raise CallRejected("invalid call", f"{name} needs exactly {sorted(fields)}, got {sorted(call)}")
    for field, check in fields.items():
        if not check(call[field]):
            raise CallRejected("invalid call", f"bad value for {field}: {call[field]!r}")
    return name, call


def check_permission(name, args, granted, approve):
    if name not in granted:
        raise CallRejected("permission denied", f"{name} is not granted for this request")
    if TOOLS[name].needs_approval and not approve(name, args):
        raise CallRejected("permission denied", f"a person did not approve {name} {args}")


# ---------------------------------------------------------------------------
# Part 3: running a tool that passed every check.
# ---------------------------------------------------------------------------

class ToolRunner:
    def __init__(self, facts=None):
        self.facts = dict(load_facts() if facts is None else facts)

    def run(self, name, args):
        if name == "add":
            return str(args["a"] + args["b"])
        if args["subject"] not in self.facts:
            raise CallRejected("not found", f"no fact about {args['subject']!r}")
        if name == "lookup":
            return self.facts[args["subject"]]
        del self.facts[args["subject"]]
        return "deleted"


def clean_result(text):
    """Tool output is untrusted data. One line only, so it can't pretend to be a new tool call line."""
    return " ".join(text.split())[:MAX_RESULT_CHARS]
