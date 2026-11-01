"""Phase 6b: the harness, the application loop around the model.

    user text -> model writes a line
        a line starting with "{" is a PROPOSED tool call:
            validate -> check permission -> run -> append "result ..." or "error ..." -> model again
        anything else is the final answer

The model is just a function from text to text, so the same harness works with the trained
model and with scripted "misbehaving" models used to test the checks.
"""
from dataclasses import dataclass, field

import tools

MAX_TOOL_CALLS = 3
MAX_INPUT_CHARS = 100


@dataclass
class Outcome:
    status: str                                    # answered | stopped | rejected_input | empty_output
    answer: str = ""
    transcript: str = ""
    log: list = field(default_factory=list)        # one entry per proposal: what happened and why


def never_approve(name, args):
    return False


def run_request(model, user_text, runner, granted=tools.READ_ONLY, approve=never_approve,
                max_tool_calls=MAX_TOOL_CALLS):
    if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > MAX_INPUT_CHARS:
        return Outcome("rejected_input")
    user_text = " ".join(user_text.split())

    transcript, log, calls = user_text + "\n", [], 0
    while True:
        line = model(transcript).strip()
        if not line:
            return Outcome("empty_output", transcript=transcript, log=log)
        if not line.startswith("{"):
            return Outcome("answered", answer=line, transcript=transcript + line + "\n", log=log)
        if calls == max_tool_calls:
            log.append({"proposal": line, "decision": "stopped", "detail": f"limit of {max_tool_calls} tool calls"})
            return Outcome("stopped", transcript=transcript, log=log)
        calls += 1
        try:
            name, args = tools.parse_call(line)
            tools.check_permission(name, args, granted, approve)
            feedback = "result " + tools.clean_result(runner.run(name, args))
            log.append({"proposal": line, "decision": "executed", "detail": feedback})
        except tools.CallRejected as rejected:
            feedback = "error " + rejected.public
            log.append({"proposal": line, "decision": "rejected", "detail": rejected.detail})
        transcript += line + "\n" + feedback + "\n"
