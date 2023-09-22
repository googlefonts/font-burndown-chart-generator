# `burndown-chart-generator` - a progress burndown chart generator script

## What it does

The script generates a chart showing completed glyphs over time.
It does this by reading color mark info out of UFO sources, using git to see what's changed as you've been working.
Milestones can be added to the chart for ease of tracking progress towards a goal.

## Setup

The main thing you need to configure in order to be able to use the burndown chart generator is its config file.
You can see the example one [here](./burndown.example.toml), which you should copy and modify to your liking.
The expected name for the file is `burndown.toml`, but if you want something different that's fine.

### Configuration TOML

`[config]` fields:

- `repo_path` (string, defaults to same folder as config file): the main folder for your project, where Git lives (run `git rev-parse --show-toplevel` if you're not sure)
- `commit_start` (string): the git revision to start the chart at (i.e. the first commit to analyse)
- `commit_end` (string): the git revision to end the chart at, typically a branch name
- `ufo_finder`: specifies the algorithm to use to find your UFOs. There are currently three options, some requiring additional fields (see the example for syntax)
  - `glob`: searches the `repo_path` for any folders with the UFO extension
  - `designspace`: reads UFO paths from a given designspace file
  - `google-fonts`: reads the [Google Fonts Tools](https://github.com/googlefonts/gftools) configuration YAML file to get paths to UFOs/designspaces
- `cache` (bool, default `false`): whether or not to save cache information to speed up subsequent runs
- `cache_path` (string, default `.burndown-chart-generator-cache.toml`): where to save cache information. Has no effect unless `cache` is `true`

#### Statuses

A status is the definition of the meaning of a given mark color & glyph type pair.
For example, a green mark on a drawn glyph should be labelled as "Ready for review (drawn)".

`[[status]]` fields:

- `name` (string): the label for glyphs in this category (shown in the legend of the chart)
- `plot_color` (string): the hex color (with `#` prefix) of the area of the chart for these glyphs
- `glyph_type` (string, optional): the type of glyph. Currently "drawn" and "composite" are supported
- `mark_color` (string or list): the color of the mark. You can give a list of 4 values between 0 and 1 (see [the UFO convention](https://unifiedfontobject.org/versions/ufo3/conventions/#colors)), or use one of the following pre-configured colors:
  - `red`
  - `yellow`
  - `green`
  - `blue`
  - `purple`

You can omit `glyph_type` and/or `mark_color` to give yourself catch-all options.

#### Milestones

A milestone is a trendline to reach number of glyphs across a number of UFOs by a certain date.
Trendlines will start at zero glyphs unless `starts_from_previous` is set to `True`.
The y axis "Number of glyph sources" is calculated by multiplying `total_glyphs` and `total_ufos`.

`[[milestone]]` fields:

- `name` (string): the name of the milestone (shown at the end of the trendline)
- `plot_color` (string): the hex color (with `#` prefix) to drawn the line
- `start_date` (YYYY-MM-DD date, optional): the start date for the target. Can be excluded if `starts_from_previous` is `True`
- `starts_from_previous` (bool, default `false`): indicates whether to start the current milestone from the previous one. Can be excluded if `start_date` is set
- `due_date` (YYYY-MM-DD date): the deadline for the milestone (where the line will end on the x axis)
- `total_glyphs` (integer): the number of glyphs expected to be in a single master when the milestone is complete
- `total_ufos` (integer): the number of masters/UFOs there are expected to be when the milestone is complete

#### Glyph types

The `[glyph_types]` section maps glyph names to their type, to help with status classification.
Glyphs can either be "drawn" or "composite"

To save the tedium of writing this by hand, you can generate it from a UFO using `burndown-chart-generator generate-glyph-types <UFO path>`, and then copy-paste it into your `burndown.toml`

## Running the tool

### On your local machine

Run `pip install burndown-chart-generator` (or add to your project's requirements/dependencies)

If your configuration file is called `burndown.toml`, then just call `burndown-chart-generator` in the same folder as it.
Otherwise, call `burndown-chart-generator --config <path to config.toml>`.

The image will be saved into the current working directory with the name `burndown-chart.png`

```
usage: burndown-chart-generator [-h] [-V] [-c CONFIG] {generate-glyph-types,export-env} ...

a font project burndown chart generator

options:
  -h, --help            show this help message and exit
  -V, --version         print the program's version and exit
  -c CONFIG, --config CONFIG
                        the path to the burndown generator config TOML file (defaults to ./burndown.toml)

subcommands:
  generate-glyph-types  generates the [glyph-types] table for config TOML file based on a UFO
  export-env            prints environment variable declarations to STDOUT. Useful for CI
```

### In GitHub Actions

See [action-burndown](https://github.com/daltonmaag/action-burndown) for this use case.
In short though, something like this:

```yml
steps:
  - name: "Checkout repo"
    uses: actions/checkout@v3
    with:
      # This does a full checkout of the repositry, making all
      # commits/branches/tags available, which is needed for generating
      # the burndown chart, as the tool traverses Git history
      fetch-depth: 0
  - name: "Generate burndown chart"
    uses: daltonmaag/action-burndown@v0.1
```

## How it works

For each commit in the specified range\*:

1. Create a [git worktree](https://git-scm.com/docs/git-worktree)
2. Count the number of glyphs with each recognised status

Then, using the status counts along with the commit dates, plots a chart to show progress

(\* if there are multiple commits in a single calendar day, only the last one is considered, as a time saving measure)

## Versioning

This project adheres to [semantic versioning](https://semver.org/)

Breaking changes that would require a major version increment:

* Making a config that was valid (passes validation **and** has useful behaviour) in a previous version no longer load
  * Changing the names of fields (without supporting fallback to the old names)
  * Adding new required fields
* Removing or renaming CLI flags
* Changing the default settings to be semantically different
  * e.g. making `starts_from_previous` default to `true` instead of `false`
* Changing the default graph file name

Notably this does not cover the appearance of the chart 

Minor version changes:

* Adding new, optional CLI flags or config file items
* Adding a new valid value to an existing flag/option
  * e.g. adding a new `ufo_finder` algorithm
