# Releasing

TLDR: before running the Release workflow, do the chores below and merge
them into the branch being released. The workflow builds from that branch,
including the user's guide PDF, so anything merged afterwards misses the
release.

## Chores, in order

1. Speed sweep, if the release has significant modem or ARQ changes:
   [speedtests/README.md](speedtests/README.md#re-running-the-sweep).
2. User's guide: every release. Recipe below.
3. Metainfo `<releases>` entry, if the release deserves one
   (`tools/bump_version.sh` leaves it alone; it needs a hand-written
   description).
4. Draft the GitHub release with its notes and run the Release workflow
   (steps in the header of `.github/workflows/release.yml`). The workflow
   bumps the four version files, not the guide.

## Updating the user's guide

The guide is `docs/user-guide.md`, its screenshots are
`docs/user-guide-img/`. Text changes go up as a PR for the maintainer's
review, apart from code changes.

### 1. Survey what changed

`PREV` is the last release tag, `NEW` the version being released:

    git fetch origin --tags
    PREV=$(git describe --tags --abbrev=0 origin/master)
    git log --first-parent --format='%h %s' $PREV..origin/master

The merged PR titles are the list to work from. Then the paths the guide describes:

    git diff --stat $PREV..origin/master -- native/gui native/web     # screenshots, sections 3, 7, 8
    git diff $PREV..origin/master -- native/app/options.cpp           # flags: sections 3, 8 "Command line only"
    git diff --stat $PREV..origin/master -- native/apps tools/package_app.sh tools/make_installer.sh
                                                                      # monitor, audio check, downloads: sections 2, 9

For each user-visible change, decide: new text, changed text, or nothing
(internal changes and receiver tuning usually need nothing, though a
change to what the user sees on screen does).

### 2. Version number

The version appears in the header line and in the download filenames.
Check every hit is the version, then replace:

    grep -n "${PREV#v}" docs/user-guide.md
    sed -i "s/${PREV#v}/$NEW/g" docs/user-guide.md

Escape the dots if the grep shows a hit that isn't the version.

### 3. Screenshots

Retake a screenshot when the survey shows a change to what it shows:
a new or moved widget, a renamed setting, a new tab. Retake all of one
kind (GUI or web) together so they match. Keep the sizes so the PDF
layout holds:

| File | Size (px) |
|---|---|
| `gui-main.png` | 877 x 560 |
| `gui-monitor.png` | 780 x 560 |
| `gui-settings-rig.png`, `gui-settings-station.png` | 623 x 900 |
| `web-monitor.png`, `web-settings.png` | 1200 x 940 |

The pictures show two stations, W1AW and K2XYZ, joined by a simulated
channel with noise and fading, mid-file-transfer (section 7 says so; keep
it true). Each station runs with `--audio-io pipe:IN,OUT` on named pipes
(`mkfifo`), with a relay between them that applies
`data2g.hfchannel.apply_channel`. The 0.5.0 relay was not committed, so
the first retake writes one; commit it under `scripts/` and update this
paragraph.

### 4. Text

- New features: add them where an operator would look (the tour of the
  screen, the settings reference, troubleshooting), not as a changelog.
- Changed behaviour: fix the existing text; don't append a note.
- New command-line flags: section 8, "Command line only".
- Removed features: delete their text.
- The contents list at the top is hand-maintained: update it with new
  sections.

### 5. Check the PDF

    docs/user-guide-pdf.sh /tmp/guide.pdf

Look at it: screenshots placed, no table overflowing the page. The script
header says how to run it without a local pandoc.

### 6. Merge

Open the PR, get it reviewed, merge, then run the Release workflow.
