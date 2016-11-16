# sirota_vr_scripts
Command-line utility programs for scanning, calibrating, and debugging the ratCAVE Virtual Reality system.

Once installed, all programs can be run from the command line (see examples below).
Almost all programs require a Motive Project File (.ttp) as a first argument.
Help text for each program is available throught the "--help" option.

## Installation

Download the source from github and install from the setup.py file:

```bash
cd ratcave_calibration
python setup.py install
```

## Examples

### Debugging: Get Rigid Body Live Tracking Info

Verify that you can track something using MotivePy!  This program will stream the 3D coordinates of a rigid body to the stdout:

```bash
track_rigidbody my_project_file.ttp
```

### Debugging: Generate a Test Dots Image

The arena scanning and projector calibration scripts require that the cameras be able to track white dots projected
onto the arena's surface.  The Motive gui is useful for verifying the correct settings, and this program will simply
put a fullscreen dot pattern up to make debugging simpler.

```bash
dots_test
```

### Arena Scanning: Collect a Point Cloud of your Arena, and Meshify It

```bash
scan_arena motive_project.ttp my_arena.obj --nsides 5 --body Arena
```

### Projector Calibration: Map 2D Projection points and 3D Marker positions into a 3D Projector Location

```bash
calibrate_projector motive_project.ttp
```