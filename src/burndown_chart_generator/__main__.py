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
from __future__ import annotations

import attrs
import cattrs
import colorsys
import os
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as Date, datetime as DateTime
from glob import glob
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import toml
import yaml
from fontTools.designspaceLib import DesignSpaceDocument  # pyright: ignore
from ufoLib2.objects import Font, Glyph

from .glyph_types_generator import print_glyph_types_for
from . import __version__ as VERSION


SUPPORTED_GLYPH_TYPES: set[str] = {"drawn", "composite"}


@dataclass
class Config:
    repo_path: Path
    caching: bool
    cache_path: Path
    git_rev_since: str
    git_rev_current: str
    glyph_types: dict[str, GlyphType]
    _ufo_finder: Callable[[Path], Iterator[Path]]
    _ufo_finder_relative_path: Path
    """Given the Git root folder in which the project has been checked out,
    returns a list of UFOs in which to look for glyph statuses.
    """
    statuses: Sequence[Status]
    """Statuses ordered from most done to least done."""
    milestones: Sequence[Milestone]

    @classmethod
    def from_file(cls, path: Path):
        V = TypeVar("V")

        def get_section(config: dict[str, Any], header: str) -> Any:
            try:
                return config[header]
            except KeyError:
                raise KeyError(f"missing [{header}] from config")

        def get(
            config_section: dict[str, Any],
            key: str,
            *,
            type_check: Optional[Type[V]] = None,
            context: Optional[str] = None,
            default: Optional[V] = None,
            optional: bool = False,
        ) -> V:
            try:
                value = config_section[key]
                if type_check is None or isinstance(value, type_check):
                    return value
                else:
                    error = f'incorrect type for "{key}"'
                    if context:
                        error += f" (section {context})"
                    error += f" in {path}: wanted {type_check.__name__}, got {type(value).__name__}"
                    raise ValueError(error)
            except KeyError:
                if default is not None or optional:
                    return default  # type: ignore
                else:
                    error = f"missing {key} from config"
                    if context:
                        error += f" (section {context})"
                    raise KeyError(error)

        raw = toml.load(path)

        raw_config = get_section(raw, "config")
        repo_path = Path(
            get(
                raw_config,
                "repo_path",
                type_check=str,
                context="[config]",
                default=path.parent,
            )
        )
        if not repo_path.is_dir():
            raise ValueError("repo_path should be a folder")
        caching = (
            get(raw_config, "cache", type_check=bool, default=False)
            or "BURNDOWN_CACHING" in os.environ
        )
        cache_path = Path(
            get(
                raw_config,
                "cache_path",
                type_check=str,
                default=f"{os.getcwd()}{os.pathsep}.burndown-chart-generator-cache.toml",
            )
        )

        glyph_types = get_section(raw, "glyph_types")
        if not set(glyph_types.values()).issubset(SUPPORTED_GLYPH_TYPES):
            raise ValueError(
                "unsupported glyph type: only drawn & composite are allowed"
            )

        ufo_finder_raw: dict[str, str] = get(
            raw_config, "ufo_finder", context="[config]"
        )
        algorithm = get(ufo_finder_raw, "algorithm", type_check=str, context="[config]")
        ufo_finder = None
        ufo_finder_relative_path = None
        if algorithm == "glob":
            ufo_finder = glob_finder
            ufo_finder_relative_path = Path(".")
        elif algorithm == "designspace":
            ufo_finder = designspace_finder
            try:
                ufo_finder_relative_path = Path(
                    ufo_finder_raw["designspace"]
                ).relative_to(repo_path)
            except ValueError:
                raise ValueError("Designspace file must be within the repository")
        elif algorithm == "google-fonts":
            ufo_finder = google_fonts_config_finder
            try:
                ufo_finder_relative_path = Path(ufo_finder_raw["config"]).relative_to(
                    repo_path
                )
            except ValueError:
                raise ValueError(
                    "Google Fonts config file must be within the repository"
                )
        else:
            raise ValueError(f'unsupported ufo_finder algorithm "{algorithm}"')

        statuses = [
            Status(
                name=get(status, "name", type_check=str, context="[[status]]"),
                glyph_type=get(status, "glyph_type", type_check=str, optional=True),  # type: ignore
                plot_color=get(
                    status, "plot_color", type_check=str, context="[[status]]"
                ),
                mark_color=get(status, "mark_color", optional=True),
            )
            for status in get_section(raw, "status")
        ]

        milestones = [
            Milestone(
                name=get(milestone, "name", type_check=str, context="[[milestone]]"),
                plot_color=get(
                    milestone, "plot_color", type_check=str, context="[[milestone]]"
                ),
                starts_from_previous=get(
                    milestone, "starts_from_previous", type_check=bool, default=False
                ),
                due_date=get(
                    milestone, "due_date", type_check=Date, context="[[milestone]]"
                ),
                total_glyphs=get(
                    milestone, "total_glyphs", type_check=int, context="[[milestone]]"
                ),
                total_ufos=get(milestone, "total_ufos", type_check=int, default=1),
            )
            for milestone in get_section(raw, "milestone")
        ]
        if len(milestones) > 0 and milestones[0].starts_from_previous:
            raise ValueError("first milestone can't start from previous")

        return cls(
            repo_path=repo_path,
            caching=caching,
            cache_path=cache_path,
            git_rev_since=get(
                raw_config, "commit_start", type_check=str, context="[config]"
            ),
            git_rev_current=get(
                raw_config, "commit_end", type_check=str, context="[config]"
            ),
            glyph_types=glyph_types,
            _ufo_finder=ufo_finder,
            _ufo_finder_relative_path=ufo_finder_relative_path,
            statuses=statuses,
            milestones=milestones,
        )

    def ufo_finder(self, within: Path) -> Iterator[Path]:
        return self._ufo_finder(within / self._ufo_finder_relative_path)

    def export_env(self) -> None:
        print(f"BCG_VERSION={VERSION}")
        print(f"BCG_REPO_PATH={self.repo_path.absolute()}")
        if self.caching:
            print("BCG_CACHE=1")
            print(f"BCG_CACHE_PATH={self.cache_path.absolute()}")
        print(f"BCG_GIT_REV_SINCE={self.git_rev_since}")
        print(f"BCG_GIT_REV_CURRENT={self.git_rev_current}")


GlyphType = Literal["drawn", "composite"]
SimpleColor = Literal["red", "yellow", "green", "blue", "purple"]
MarkColor = Union[SimpleColor, Tuple[float, float, float, float]]


def structure_mark_color(data, _type) -> Any:
    if data is None or isinstance(data, (str, tuple)):
        return data
    else:
        raise TypeError(f"invalid mark color: {data}")


cattrs.register_structure_hook(Optional[MarkColor], structure_mark_color)


@attrs.frozen(kw_only=True)
class Status:
    name: str
    plot_color: str
    glyph_type: Optional[GlyphType] = None
    mark_color: Optional[MarkColor] = None
    lib_key_name: Optional[str] = None
    lib_key_value: Optional[Any] = None


@dataclass
class Milestone:
    name: str
    plot_color: str
    due_date: Date
    total_glyphs: int
    total_ufos: int
    start_date: Optional[Date] = None
    starts_from_previous: bool = False


@dataclass
class Repo:
    path: Path

    def git(self, *args: str, check: bool = True) -> str:
        command = ["git", "-C", str(self.path), *args]
        # print(f"Running {' '.join(command)}")
        res = run(command, check=check, capture_output=True, encoding="utf-8")
        return res.stdout


@dataclass
class Revision:
    sha: str
    date: Date
    _repo: Repo

    @contextmanager
    def checkout(self):
        try:
            with TemporaryDirectory() as tmpdir:
                self._repo.git("worktree", "add", "--detach", tmpdir, self.sha)
                yield Path(tmpdir)
        finally:
            self._repo.git("worktree", "remove", tmpdir, check=False)


# region UFO finders
def glob_finder(root: Path) -> Iterator[Path]:
    for path in glob(f"{root}/**/*.ufo"):
        yield Path(path)


def designspace_finder(designspace_path: Path) -> Iterator[Path]:
    designspace: Any = DesignSpaceDocument.fromfile(designspace_path)
    for source in designspace.sources:
        # Exclude sparse sources
        if source.path and source.layerName == None:
            yield Path(source.path)


def google_fonts_config_finder(config_path: Path) -> Iterator[Path]:
    config: Any = yaml.load(config_path.read_text())  # type: ignore
    for source_path in config["sources"]:
        source_path = Path(source_path)
        if source_path.suffix == "designspace":
            for ufo_path in designspace_finder(source_path):
                yield ufo_path
        elif source_path.suffix == "ufo":
            yield source_path
        else:
            print(f"WARNING: unsupported source type: {source_path}")


# endregion


def iter_revisions(
    repo_path: Path, rev_since: str, rev_current: str
) -> Iterator[Revision]:
    """Iterate through the given git revisions, and for each checkout the
    repository into a temp folder and yield that, along with the date of the
    revision.
    """
    repo = Repo(repo_path)
    out = repo.git("rev-list", "--format=tformat:%H %aI", f"{rev_since}..{rev_current}")
    lines = [line for line in out.splitlines() if not line.startswith("commit")]

    all_commits: list[Tuple[DateTime, str]] = []
    for line in lines:
        sha, date_iso = line.split(maxsplit=1)
        date = DateTime.fromisoformat(date_iso)
        all_commits.append((date, sha))

    # Process only the last commit of each day, in case of several commits per day.
    filtered_commits: list[Tuple[Date, str]] = []
    for date_time, sha in sorted(all_commits):
        if len(filtered_commits) > 0 and date_time.date() == filtered_commits[-1][0]:
            # Same day, replace with this one which is later in the day
            filtered_commits[-1] = (date_time.date(), sha)
        else:
            filtered_commits.append((date_time.date(), sha))

    for i, (date, sha) in enumerate(filtered_commits):
        print(f"Processing commit {i+1}/{len(filtered_commits)}: {sha} on {date}")
        yield Revision(sha, date, repo)


# region Glyph processing
def glyph_matches_status(
    glyph_types: dict[str, str], glyph: Glyph, status: Status
) -> bool:
    return (
        glyph_matches_type(glyph_types, glyph, status)
        and glyph_matches_color(glyph, status)
        and glyph_matches_lib_key(glyph, status)
    )


def glyph_matches_type(
    glyph_types: dict[str, str], glyph: Glyph, status: Status
) -> bool:
    if status.glyph_type is None:
        return True
    assert glyph.name
    return status.glyph_type == glyph_types.get(glyph.name, None)


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
    counts_by_date: Mapping[Date, Sequence[int]],
    image_path: Path,
):
    # Example code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/stackplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-stackplot-demo-py
    dates: list[Date] = []
    counts_by_status: list[list[int]] = [[] for _ in config.statuses]
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
    ax.set_title(
        f"{config.repo_path.absolute().stem} on {config.git_rev_current} since {config.git_rev_since}"
    )
    ax.set_xlabel("Commit date")
    ax.set_ylabel("Number of glyph sources")
    ax.tick_params(axis="x", labelrotation=50)

    fig.tight_layout(pad=3)
    fig.savefig(str(image_path))


# region Color processing
# https://en.wikipedia.org/wiki/Hue#24_hues_of_HSL/HSV
HUE_TO_COLOR: list[Tuple[int, SimpleColor]] = [
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
@attrs.define(kw_only=True)
class Cache:
    # All fields that aren't inner are cache keys
    _statuses: set[Status]
    _glyph_types: dict[str, GlyphType]
    _burndown_chart_generator_version: str = attrs.field(default=VERSION)
    _inner: dict[str, list[int]] = attrs.field(default=defaultdict(list))

    def __contains__(self, key: str) -> bool:
        return key in self._inner

    def __getitem__(self, key: str) -> List[int]:
        return self._inner[key]

    def __setitem__(self, key: str, value: list[int]):
        self._inner[key] = value

    @classmethod
    def new(cls, config: Config) -> Cache:
        return cls(
            statuses=frozenset(config.statuses),
            glyph_types=config.glyph_types,
        )

    @classmethod
    def from_file(cls, cache_path: Path, config: Config) -> Cache:
        if cache_path.exists():
            print("Loading cache")
            unstructured = toml.load(cache_path)
            try:
                converter = cattrs.Converter()
                cache = cattrs.structure(unstructured, cls)
                if cache.matches(config):
                    print(
                        "Ignoring cache (for an outdated version for burndown-chart-generator)"
                    )
                    return cache
                else:
                    return Cache.new(config)
            except Exception as cattrs_err:
                print("ERROR: unable to load cache")
                print(cattrs.transform_error(cattrs_err))
                return Cache.new(config)
        else:
            return Cache.new(config)

    def save(self, cache_path: Path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        converter = cattrs.Converter(
            unstruct_collection_overrides={
                Set: list,
            }
        )
        unstructured = converter.unstructure(self)
        cache_path.write_text(
            f"# This is a @generated file, do not modify{os.linesep}{toml.dumps(unstructured)}"
        )

    def matches(self, config: Config):
        return (
            # TODO: follow semver
            self._burndown_chart_generator_version == VERSION
            and self._statuses == set(config.statuses)
            and self._glyph_types == config.glyph_types
        )


# endregion


def main(config_path: Path) -> None:
    config = Config.from_file(config_path)
    cache = (
        Cache.from_file(config.cache_path, config)
        if config.caching
        else Cache.new(config)
    )

    counts_by_date: dict[Date, list[int]] = defaultdict(
        lambda: [0 for _ in config.statuses]
    )

    # Data just for testing the graph
    # counts_by_date = {
    #     Date(2022, 12, 1): [0, 500, 500, 3000, 1000, 30000, 30000],
    #     Date(2022, 12, 10): [500, 500, 10000, 10000, 10000, 20000, 20000],
    #     Date(2022, 12, 20): [10000, 1000, 10000, 10000, 30000, 5000, 5000],
    #     Date(2022, 12, 30): [30000, 11000, 5000, 5000, 10000, 0, 0],
    # }

    print("Preparing git worktree")
    for revision in iter_revisions(
        config.repo_path, config.git_rev_since, config.git_rev_current
    ):
        if revision.sha in cache:
            print("Using cached entry")
            value = cache[revision.sha]
            # empty list is unhelpful to add, we'd rather just rely on the defaultdict
            # in that case
            if len(value) > 0:
                counts_by_date[revision.date] = value
        else:
            counts = []
            with revision.checkout() as tmpdir:
                print("Opening UFOs")
                for ufo_path in config.ufo_finder(tmpdir):
                    try:
                        ufo = Font.open(ufo_path)
                    except Exception as e:
                        relative_path = ufo_path.relative_to(tmpdir)
                        print(f"\nReading UFO '{relative_path}' failed, skipping: {e}")
                        continue
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
                            if glyph_matches_status(config.glyph_types, glyph, status):
                                counts[i] += 1
                                break
            cache[revision.sha] = counts
            print(" done")

    output_path = Path(".") / "burndown-chart.png"
    print(f"Writing out {output_path}")
    if config.caching:
        print(f"Writing cache {config.cache_path}")
        cache.save(config.cache_path)
    plot_to_image(config, counts_by_date, output_path)


def clap() -> None:
    parser = ArgumentParser(description="a font project burndown chart generator")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"v{VERSION}",
        help="print the program's version and exit",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="the path to the burndown generator config TOML file (defaults to ./burndown.toml)",
        default=Path("burndown.toml"),
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
    )
    glyph_types_subcommand = subparsers.add_parser(
        "generate-glyph-types",
        help="generates the [glyph-types] table for config TOML file based on a UFO",
    )
    glyph_types_subcommand.add_argument(
        "ufo",
        type=Path,
        help="the path to the UFO to generate the config from",
    )
    export_env_subcommand = subparsers.add_parser(
        "export-env",
        help="prints environment variable declarations to STDOUT. Useful for CI",
    )
    export_env_subcommand.add_argument(
        "-c",
        "--config",
        type=Path,
        help="the path to the burndown generator config TOML file (defaults to ./burndown.toml)",
        default=Path("burndown.toml"),
    )

    args = parser.parse_args()
    match args.subcommand:
        case None:
            main(args.config)
        case "generate-glyph-types":
            print_glyph_types_for(args.ufo)
        case "export-env":
            Config.from_file(args.config).export_env()


if __name__ == "__main__":
    clap()
