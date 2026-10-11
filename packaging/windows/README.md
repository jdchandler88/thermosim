# Installing and launching ThermoSim on Windows

ThermoSim is a Python program. You do not need to install Python yourself: a free
tool called **uv** downloads the right Python and all the libraries for you.

There are two ways to set it up. Pick one.

## Option A: run it from a copy of the repository (simplest)

1. **Install uv.** Open *PowerShell* (press the Windows key, type `PowerShell`, press
   Enter) and run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Close PowerShell and open it again, then check it worked:

   ```powershell
   uv --version
   ```

2. **Get ThermoSim.** Either install [Git for Windows](https://git-scm.com/download/win)
   and run

   ```powershell
   git clone https://github.com/jdchandler88/thermosim
   ```

   or open <https://github.com/jdchandler88/thermosim>, press *Code*, then
   *Download ZIP*, and unzip it somewhere (for example `C:\Users\you\thermosim`).

3. **Launch it.** Double-click `packaging\windows\thermosim.bat` in that folder. The
   first launch downloads Python and the libraries (a few hundred MB, one to two
   minutes). After that it opens in a few seconds. Or, from PowerShell in the folder:

   ```powershell
   uv run thermosim gui
   ```

   To make a desktop shortcut, right-click `thermosim.bat`, choose *Show more options*,
   *Send to*, *Desktop (create shortcut)*.

4. **Update later:** `git pull` in the folder (or download the ZIP again).

## Option B: install it as a program

1. Install uv as in step 1 above.
2. Install ThermoSim (needs Git for Windows):

   ```powershell
   uv tool install git+https://github.com/jdchandler88/thermosim
   ```

3. Launch it from any PowerShell window with `thermosim gui`, or double-click
   `thermosim.bat` (it finds the installed program).
4. **Update later:** `uv tool upgrade thermosim`. **Remove:** `uv tool uninstall thermosim`.

## First run

- The first time you press *Run*, a small window says **"Preparing solver (first run
  only)..."**. The simulation core is compiled once (10 to 60 seconds) and then
  remembered.
- Use *File > Examples* to open a ready-made setup, press *Run (Fast)*, and a run of
  about 20 minutes of simulated time finishes in well under a minute.
- The program remembers your unit choice (US or metric) between launches.
- If Windows SmartScreen or your antivirus complains about `uv` or Python the first
  time, that is the downloaded Python being scanned; allow it.

See the [user guide](../../docs/user-guide.md) for how to use the program.

## Command line

The same installation provides the `thermosim` command, for example:

```powershell
uv run thermosim examples                          # list the example setups
uv run thermosim run my_setup.toml --out results   # run without the GUI
uv run thermosim validate my_setup.toml            # check a setup, estimate run time
```

## Troubleshooting

| Problem | What to do |
|---|---|
| `uv` is not recognised | Close and reopen PowerShell after installing uv. |
| The window is blurry or too big on a high-DPI screen | Change the Windows display scale (Settings > Display > Scale); ThermoSim follows it. |
| A very slow run | Use a larger cell size (10 mm is a quick preview) in the *Simulation* tab. |
| "Cannot run: ..." message | The message names the field that is wrong; it is highlighted in red in the setup. |

## Manual test checklist (for a new Windows machine)

- [ ] `uv --version` prints a version.
- [ ] `thermosim.bat` (or `uv run thermosim gui`) opens the window.
- [ ] *File > Examples > small pot stove*, then *Run (Fast)*: the progress dialog appears,
      the run finishes with "Target reached", the slice and chart are filled in.
- [ ] *Run (Real-time)*, speed 10x: moving the power slider changes the power trace.
- [ ] *File > Save run...*, then *File > Open run...* shows the same results.
- [ ] *Export CSV* writes a file that opens in Excel.
