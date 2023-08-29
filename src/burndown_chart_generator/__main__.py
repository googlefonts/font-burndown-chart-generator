# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a sort of burndown chart to evaluate progress.

How to run:
* create and activate a venv with matplotlib and pyyaml
* run `python scripts/burndown-chart-generator.py`

How to tweak: edit the `CONFIG = Config(...)` object below.
"""


from __future__ import annotations

import colorsys
import hashlib
import json
import sys
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from glob import glob

import matplotlib.pyplot as plt
from fontTools.designspaceLib import DesignSpaceDocument
from ufoLib2.objects import Font, Glyph


@dataclass
class Config:
    repo_path: Path
    git_rev_since: str
    git_rev_current: str
    ufo_finder: Callable[[Path], List[Path]]
    """Given the Git root folder in which the project has been checked out,
    returns a list of UFOs in which to look for glyph statuses.
    """
    statuses: Sequence[Status]
    """Statuses ordered from most done to least done."""
    milestones: Sequence[Milestone]


GlyphType = Literal["drawn", "composite"]
SimpleColor = Literal["red", "yellow", "green", "blue", "purple"]


@dataclass
class Status:
    name: str
    plot_color: str
    glyph_type: Optional[GlyphType] = None
    mark_color: Optional[Union[SimpleColor, Tuple[float, float, float, float]]] = None
    lib_key_name: Optional[str] = None
    lib_key_value: Optional[Any] = None


@dataclass
class Milestone:
    name: str
    plot_color: str
    due_date: datetime
    total_glyphs: int
    total_ufos: int
    start_date: Optional[datetime] = None
    starts_from_previous: bool = False


@dataclass
class Repo:
    path: Path

    def git(self, *args: str, check=True) -> str:
        command = ["git", "-C", str(self.path), *args]
        # print(f"Running {' '.join(command)}")
        res = run(command, check=check, capture_output=True, encoding="utf-8")
        return res.stdout


@dataclass
class Revision:
    sha: str
    date: datetime
    _repo: Repo

    @contextmanager
    def checkout(self):
        try:
            with TemporaryDirectory() as tmpdir:
                self._repo.git("worktree", "add", "--detach", tmpdir, self.sha)
                yield Path(tmpdir)
        finally:
            self._repo.git("worktree", "remove", tmpdir, check=False)


# Small helper functions to count UFOs; they just return the number of args
def _count(*args):
    return len(args)


opsz = wdth = wght = ROND = GRAD = ital = _count


def iter_revisions(repo_path, rev_since, rev_current):
    """Iterate through the given git revisions, and for each checkout the
    repository into a temp folder and yield that, along with the date of the
    revision.
    """
    repo = Repo(repo_path)
    out = repo.git("rev-list", "--format=tformat:%H %aI", f"{rev_since}..{rev_current}")
    lines = [line for line in out.splitlines() if not line.startswith("commit")]

    all_dates_and_shas = []
    for line in lines:
        sha, date_iso = line.split(maxsplit=1)
        date = datetime.fromisoformat(date_iso)
        all_dates_and_shas.append((date, sha))

    # Process only the last commit of each day, in case of several commits per day.
    dates_and_shas = []
    for date, sha in sorted(all_dates_and_shas):
        if dates_and_shas and date.date() == dates_and_shas[-1][0].date():
            # Same day, replace with this one which is later in the day
            dates_and_shas[-1] = (date, sha)
        else:
            dates_and_shas.append((date, sha))

    for i, (date, sha) in enumerate(dates_and_shas):
        print(f"Processing commit {i+1}/{len(dates_and_shas)}: {sha} on {date}")
        yield Revision(sha, date, repo)


# region Glyph processing
def glyph_matches_status(glyph: Glyph, status: Status) -> bool:
    return (
        glyph_matches_type(glyph, status)
        and glyph_matches_color(glyph, status)
        and glyph_matches_lib_key(glyph, status)
    )


def glyph_matches_type(glyph: Glyph, status: Status) -> bool:
    if status.glyph_type is None:
        return True
    return status.glyph_type == GLYPH_TYPES.get(glyph.name, None)


def glyph_matches_color(glyph: Glyph, status: Status) -> bool:
    if status.mark_color is None:
        return True
    if glyph.markColor is None:
        return False
    r, g, b, _a = parse_mark_color(glyph.markColor)
    if isinstance(status.mark_color, str):
        return describe_color(r, g, b) == status.mark_color
    else:
        return (r, g, b) == status.mark_color[:3]


def glyph_matches_lib_key(glyph: Glyph, status: Status) -> bool:
    if status.lib_key_name is None:
        return True
    return glyph.lib.get(status.lib_key_name, None) == status.lib_key_value


# endregion


def plot_to_image(
    config: Config,
    counts_by_date: Mapping[datetime, Sequence[int]],
    image_path: Path,
):
    # Example code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/stackplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-stackplot-demo-py
    dates = []
    counts_by_status: List[List[int]] = [[] for _ in config.statuses]
    for date, counts in sorted(counts_by_date.items()):
        dates.append(date)
        for i, count in enumerate(counts):
            counts_by_status[i].append(count)

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.stackplot(
        dates,
        counts_by_status,
        colors=[status.plot_color for status in config.statuses],
        labels=[status.name for status in config.statuses],
    )
    for index, milestone in enumerate(config.milestones):
        if not milestone.starts_from_previous:
            ax.plot(
                [milestone.start_date, milestone.due_date],
                [0, milestone.total_glyphs * milestone.total_ufos],
                color=milestone.plot_color,
            )
        elif index == 0:
            raise IndexError("first milestone can't continue from previous")
        else:
            previous = config.milestones[index - 1]
            ax.plot(
                [previous.due_date, milestone.due_date],
                [
                    previous.total_glyphs * previous.total_ufos,
                    milestone.total_glyphs * milestone.total_ufos,
                ],
                color=milestone.plot_color,
            )
        ax.plot(
            [milestone.due_date, milestone.due_date],
            [0, milestone.total_glyphs * milestone.total_ufos],
            color=milestone.plot_color,
            linestyle="dashed",
        )
        ax.text(
            milestone.due_date,  # type: ignore
            milestone.total_glyphs * milestone.total_ufos,
            milestone.name,
            horizontalalignment="right",
            verticalalignment="bottom",
            multialignment="right",
            color=milestone.plot_color,
            bbox=dict(facecolor="#ffffffc0", edgecolor="#d6d6d6", boxstyle="round"),
        )
    ax.legend(loc="upper left")
    print
    ax.set_title(
        f"{CONFIG.repo_path.absolute().stem} on {config.git_rev_current} since {config.git_rev_since}"
    )
    ax.set_xlabel("Commit date")
    ax.set_ylabel("Number of glyph sources")
    ax.tick_params(axis="x", labelrotation=50)

    fig.tight_layout(pad=3)
    fig.savefig(str(image_path))


# region Color processing
# https://en.wikipedia.org/wiki/Hue#24_hues_of_HSL/HSV
HUE_TO_COLOR: List[Tuple[int, SimpleColor]] = [
    (30, "red"),  # Up to 30°, classify as red
    (75, "yellow"),
    (165, "green"),
    (255, "blue"),
    (315, "purple"),
    (360, "red"),
]


def parse_mark_color(color: str) -> Tuple[float, float, float, float]:
    # https://unifiedfontobject.org/versions/ufo3/conventions/#colors
    r, g, b, a = color.split(",")
    return float(r), float(g), float(b), float(a)


def describe_color(r: float, g: float, b: float) -> SimpleColor:
    h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    for degrees, color_name in HUE_TO_COLOR:
        if h <= degrees / 360.0:
            return color_name
    return HUE_TO_COLOR[-1][1]


# [(10, '0.2288,1,0.4511,1', '#3AFF73FF'),
assert describe_color(0.2288, 1, 0.4511) == "green"
#  (18, '0.9908,1,0.037,1', '#FDFF09FF'),
assert describe_color(0.9908, 1, 0.037) == "yellow"
#  (49, '0.8687,0.1142,0.999,1', '#DE1DFFFF'),
assert describe_color(0.8687, 0.1142, 0.999) == "purple"
#  (200, '1,0,0,1', '#FF0000FF'),
assert describe_color(1, 0, 0) == "red"
#  (244, '1,1,0,1', '#FFFF00FF'),
assert describe_color(1, 1, 0) == "yellow"
#  (349, '0,0,1,1', '#0000FFFF'),
assert describe_color(0, 0, 1) == "blue"
#  (1257, '0.0941,0.7922,0.9961,1', '#18CAFEFF'),
assert describe_color(0.0941, 0.7922, 0.9961) == "blue"
#  (1800, '0.884,0.8791,0.0317,1', '#E1E008FF'),
assert describe_color(0.884, 0.8791, 0.0317) == "yellow"
#  (2361, '0.1227,0.9628,0.999,1', '#1FF6FFFF'),
assert describe_color(0.1227, 0.9628, 0.999) == "blue"
#  (4020, '0.2431,0.9922,0.5255,1', '#3EFD86FF'),
assert describe_color(0.2431, 0.9922, 0.5255) == "green"
#  (4304, '0.1313,0.9997,0.0236,1', '#21FF06FF')]
assert describe_color(0.1313, 0.9997, 0.0236) == "green"
# endregion


# region Caching
# https://stackoverflow.com/a/44873382
def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def get_cache() -> Optional[dict[str, list[int]]]:
    if CACHE_PATH.exists():
        print("Loading cache")
        return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict[str, list[int]]):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


SELF_HASH = sha256(__file__)
CACHE_PATH = (
    Path(__file__).parent / ".burndown-chart-generator-cache" / f"{SELF_HASH}.json"
)
# endregion


def main() -> None:
    caching = "BURNDOWN_CACHING" in os.environ
    cache = get_cache() or {}
    config = CONFIG
    counts_by_date: Dict[datetime, List[int]] = defaultdict(
        lambda: [0 for _ in config.statuses]
    )

    # Data just for testing the graph
    # counts_by_date = {
    #     datetime(2022, 12, 1): [0, 500, 500, 3000, 1000, 30000, 30000],
    #     datetime(2022, 12, 10): [500, 500, 10000, 10000, 10000, 20000, 20000],
    #     datetime(2022, 12, 20): [10000, 1000, 10000, 10000, 30000, 5000, 5000],
    #     datetime(2022, 12, 30): [30000, 11000, 5000, 5000, 10000, 0, 0],
    # }

    print("Preparing git worktree")
    for revision in iter_revisions(
        config.repo_path, config.git_rev_since, config.git_rev_current
    ):
        if revision.sha in cache:
            print("Using cached entry")
            counts_by_date[revision.date] = cache[revision.sha]
        else:
            counts = []
            with revision.checkout() as tmpdir:
                print("Opening UFOs", end="")
                for ufo_path in config.ufo_finder(tmpdir):
                    try:
                        ufo = Font.open(ufo_path)
                    except Exception as e:
                        relative_path = ufo_path.relative_to(tmpdir)
                        print(f"\nReading UFO '{relative_path}' failed, skipping: {e}")
                        continue
                    print(".", end="", flush=True)
                    for glyph_name in ufo.keys():
                        try:
                            glyph = ufo[glyph_name]
                        except Exception as e:
                            relative_path = ufo_path.relative_to(tmpdir)
                            print(
                                f"\nReading glyph '{glyph_name}' from UFO '{relative_path}' failed, skipping: {e}"
                            )
                            continue
                        counts = counts_by_date[revision.date]
                        for i, status in enumerate(config.statuses):
                            if glyph_matches_status(glyph, status):
                                counts[i] += 1
                                break
            cache[revision.sha] = counts
            print(" done")

    output_path = Path(".") / "burndown-chart.png"
    print(f"Writing out {output_path}")
    if caching:
        print(f"Writing cache {SELF_HASH}.json")
        save_cache(cache)
    plot_to_image(config, counts_by_date, output_path)


# In IPython:
# from ufoLib2 import Font
# f = Font.open("sources/MyFont.ufo")
# print({g.name: "drawn" if g.contours else "composite" for g in f})
GLYPH_TYPES = {
    "A": "drawn",
    "Aacute": "composite",
    "Adieresis": "composite",
    "B": "drawn",
    "C": "drawn",
    "D": "drawn",
    "E": "drawn",
    "F": "drawn",
    "G": "drawn",
    "H": "drawn",
    "I": "drawn",
    "I.narrow": "drawn",
    "IJ": "drawn",
    "J": "drawn",
    "J.narrow": "drawn",
    "K": "drawn",
    "L": "drawn",
    "M": "drawn",
    "N": "drawn",
    "O": "drawn",
    "P": "drawn",
    "Q": "drawn",
    "R": "drawn",
    "S": "drawn",
    "S.closed": "drawn",
    "T": "drawn",
    "U": "drawn",
    "V": "drawn",
    "W": "drawn",
    "X": "drawn",
    "Y": "drawn",
    "Z": "drawn",
    "acute": "drawn",
    "arrowdown": "drawn",
    "arrowleft": "drawn",
    "arrowright": "drawn",
    "arrowup": "drawn",
    "colon": "composite",
    "comma": "drawn",
    "dieresis": "composite",
    "dot": "drawn",
    "period": "drawn",
    "quotedblbase": "composite",
    "quotedblleft": "composite",
    "quotedblright": "composite",
    "quotesinglbase": "composite",
    "semicolon": "composite",
    "space": "composite",
}


# region UFO finders
def find_all_ufos(root: Path) -> List[Path]:
    return [Path(path) for path in glob(f"{root}/**/*.ufo")]


# endregion

# Green - design finished and ready for review
# Yellow - in progress
# Red - not started
# Blue - in progress for a future version
CONFIG = Config(
    repo_path=Path(sys.argv[1]),
    git_rev_since="45389b8f316013ef86d83b221760e50dcff387b6",
    git_rev_current="origin/master",
    ufo_finder=find_all_ufos,
    statuses=[
        Status(
            name="Ready for review (drawn)",
            plot_color="#2ecc71",
            glyph_type="drawn",
            mark_color="green",
        ),
        Status(
            name="Ready for review (composite)",
            glyph_type="composite",
            plot_color="#a9eec6",
            mark_color="green",
        ),
        Status(
            name="In progress for v1.000 (drawn)",
            plot_color="#3498db",
            glyph_type="drawn",
            mark_color="blue",
        ),
        Status(
            name="In progress for v1.000 (composite)",
            plot_color="#aac7db",
            glyph_type="composite",
            mark_color="blue",
        ),
        Status(
            name="None of the above (drawn)",
            plot_color="#e74c3c",
            glyph_type="drawn",
        ),
        Status(
            name="None of the above (composite & unknown)",
            plot_color="#e7b4af",
            # Catch-all
            # glyph_type="composite",
        ),
    ],
    milestones=[
        Milestone(
            name="Concept",
            plot_color="#1d5c85",
            start_date=datetime(2023, 5, 27),
            due_date=datetime(2023, 6, 1),
            total_glyphs=len("HAKOGgnoakv "),
            total_ufos=1,
        ),
        Milestone(
            name="Prototype",
            plot_color="#1d5c85",
            starts_from_previous=True,
            due_date=datetime(2023, 6, 14),
            total_glyphs=len("0123457BDENPRSUWbeqstuwzˆˇ˘˛˜˝HAKOGgnoakv "),
            total_ufos=2,
        ),
        Milestone(
            name="Alpha",
            plot_color="#1d5c85",
            starts_from_previous=True,
            due_date=datetime(2023, 8, 31),
            total_glyphs=len(GLYPH_TYPES),
            total_ufos=4,
        ),
    ],
)


if __name__ == "__main__":
    main()
