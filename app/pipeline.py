"""The complete application: one request-to-response pipeline joining every phase.

    input checks (5b) -> model (4b, 6a or 6b) with sampling (5a) -> tool harness (6b) -> output checks (5b)

The web page in server.py is a thin layer over `App.respond`.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase5b_evaluation"))
sys.path.insert(0, str(ROOT / "phase6b_tools"))

import guardrails  # noqa: E402  (Phase 5b; also makes Phases 4 and 5a importable)
import harness  # noqa: E402  (Phase 6b)
import metrics  # noqa: E402
import tool_model  # noqa: E402
import tools  # noqa: E402
import torch  # noqa: E402
from sampling import bpe, generate, load_checkpoint, next_token_probs  # noqa: E402

MODELS = {
    "story": dict(label="Story model", phase="4b", path=ROOT / "phase4_transformer/checkpoints/tinygpt.pt",
                  examples=["Once upon a time", "The king and the queen", "The dragon"],
                  about="Pretrained on 27 stories. It only continues text."),
    "finetuned": dict(label="Fine-tuned model", phase="6a", path=ROOT / "phase6_finetuning/checkpoints/finetuned.pt",
                      examples=["About the sea.", "About the fox.", "About the moon."],
                      about="The story model after training on question-and-answer pairs. Right format, invented content."),
    "preference": dict(label="Preference-tuned model", phase="6a", path=ROOT / "phase6_finetuning/checkpoints/preference.pt",
                       examples=["About the owl.", "About the dragon."],
                       about="The fine-tuned model nudged towards 'happy' answers. It scored 100% on its metric; judge the answers yourself."),
    "tools": dict(label="Tool-calling model", phase="6b", path=tool_model.CHECKPOINT,
                  examples=["add 40 and 2", "about the sea", "about the moon", "delete the sea"],
                  about="Proposes actions. Your application checks and runs them."),
}


def answer_uses_result(answer, log):
    """Does the final answer contain what the last successful tool returned?"""
    executed = [e for e in log if e["decision"] == "executed"]
    return not executed or executed[-1]["detail"].removeprefix("result ").lower() in answer.lower()


class App:
    def __init__(self):
        self.models = {}
        for name, spec in MODELS.items():
            if not spec["path"].exists():
                continue
            model, tokenizer, info = load_checkpoint(spec["path"])
            known = set(metrics.words(info.get("training_text", "")))
            self.models[name] = dict(model=model, tokenizer=tokenizer, known=known, text=info.get("training_text", ""))
        self.tool_model = None
        if "tools" in self.models:
            self.tool_model = tool_model.TrainedModel(self.models["tools"]["model"], self.models["tools"]["tokenizer"])
        self.reset()

    def reset(self):
        self.runner = tools.ToolRunner()   # stored facts live in memory, so a delete lasts until reset

    def info(self):
        return {"models": [dict(id=n, label=s["label"], phase=s["phase"], about=s["about"], examples=s["examples"])
                           for n, s in MODELS.items() if n in self.models]}

    def respond(self, mode, prompt, guards=True, temperature=0.8, top_p=0.9, tokens=40, seed=None,
                grant_delete=False, approved_call=None):
        if mode not in self.models:
            raise ValueError(f"unknown or untrained model: {mode}")
        loaded = self.models[mode]
        if not prompt:
            raise ValueError("Please type something first.")   # the model can't start from nothing, guards or not
        result = dict(mode=mode, status="allowed", answer="", reasons=[], steps=[], tokens=[], next_tokens=[],
                      calls=[], approval_request=None, seed=None)
        steps = result["steps"]

        if guards:
            block, flags = guardrails.check_input(prompt, loaded["tokenizer"], loaded["known"],
                                                  loaded["model"].cfg.context_length, check_familiar=mode != "tools")
            steps.append(dict(name="Input checks", phase="5b", state="blocked" if block else "warned" if flags else "passed",
                              detail="; ".join(block + flags) or "Nothing wrong found."))
            if block:
                return dict(result, status="blocked_input", reasons=block)
            result["reasons"] += flags
        else:
            steps.append(dict(name="Input checks", phase="5b", state="off", detail="Switched off for this request."))
            flags = []

        result["tokens"] = [bpe.show_token(t) for t in loaded["tokenizer"].tokenize(prompt)]
        window = loaded["tokenizer"].encode(prompt)[-loaded["model"].cfg.context_length:]
        steps.append(dict(name="Tokenizer", phase="3a", state="passed",
                          detail=f"{len(result['tokens'])} tokens; the model sees the last {len(window)}."))

        if mode == "tools":
            return self._respond_tools(result, prompt, guards, grant_delete, approved_call, flags)

        with torch.no_grad():
            logits = loaded["model"](torch.tensor([window]))[0][0, -1]
        probs = next_token_probs(logits, temperature, top_p)
        top = torch.topk(probs, min(8, len(probs)))
        result["next_tokens"] = [dict(token=bpe.show_token(loaded["tokenizer"].vocab[i]), p=p)
                                 for p, i in zip(top.values.tolist(), top.indices.tolist()) if p > 0]

        result["seed"] = random.randrange(1_000_000) if seed is None else seed
        tuned = mode != "story"
        text = generate(loaded["model"], loaded["tokenizer"], prompt + ("\n" if tuned else ""), tokens,
                        temperature, top_p, result["seed"])
        answer = text.strip().split("\n")[0] if tuned else text
        steps.append(dict(name="Model", phase="4", state="passed",
                          detail=f"Generated up to {tokens} tokens (temperature {temperature}, top-p {top_p}, seed {result['seed']})."))
        return self._check_output(result, answer, loaded, guards, flags)

    def _check_output(self, result, answer, loaded, guards, flags):
        if not guards:
            result["steps"].append(dict(name="Output checks", phase="5b", state="off", detail="Switched off for this request."))
            return dict(result, answer=answer, status="allowed")
        block, out_flags = guardrails.check_output(answer, loaded["text"], loaded["known"] or set(metrics.words(answer)))
        result["steps"].append(dict(name="Output checks", phase="5b", state="blocked" if block else "warned" if out_flags else "passed",
                                    detail="; ".join(block + out_flags) or "Nothing wrong found."))
        if block:
            return dict(result, status="blocked_output", reasons=result["reasons"] + block)
        reasons = result["reasons"] + out_flags
        return dict(result, answer=answer, reasons=reasons, status="flagged" if reasons else "allowed")

    def _respond_tools(self, result, prompt, guards, grant_delete, approved_call, flags):
        granted = tools.READ_ONLY | {"delete"} if grant_delete else tools.READ_ONLY

        def approve(name, args):
            return approved_call == {"tool": name, **args}

        outcome = harness.run_request(self.tool_model, prompt, self.runner, granted, approve)
        result["calls"] = outcome.log
        result["transcript"] = outcome.transcript
        for entry in outcome.log:
            result["steps"].append(dict(name="Tool call", phase="6b", state="passed" if entry["decision"] == "executed" else "blocked",
                                        detail=f"{entry['proposal']} -> {entry['decision']}: {entry['detail']}"))
            if entry["decision"] == "rejected" and "did not approve" in entry["detail"]:
                result["approval_request"] = json.loads(entry["proposal"])
        if outcome.status != "answered":
            return dict(result, status=outcome.status, reasons=[f"The tool loop ended with: {outcome.status}"])

        answer = outcome.answer
        if not answer_uses_result(answer, outcome.log):
            # The model garbled a correct tool result, so show the application's own copy of it.
            source = [e for e in outcome.log if e["decision"] == "executed"][-1]["detail"].removeprefix("result ")
            result["reasons"].append(f"The model wrote {answer!r}, which doesn't match the tool result. Showing the tool result.")
            result["steps"].append(dict(name="Answer check", phase="app", state="warned", detail=f"Replaced the model's answer with the tool result: {source}"))
            answer = source
        else:
            result["steps"].append(dict(name="Answer check", phase="app", state="passed", detail="The answer matches the tool result, or no tool ran."))
        return self._check_output(result, answer, self.models["tools"], guards, flags)
