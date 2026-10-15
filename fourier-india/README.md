# Tracing India's border with Fourier epicycles

![Fourier epicycles tracing India's border with 5, 10, 15, 100 and 200 circles](preview.gif)

The border of India is a closed curve. Read it as a sequence of complex
numbers `z[n] = x[n] + i·y[n]` and take its discrete Fourier transform:

```
c[k] = (1/M) · Σ z[n] · e^(−2πi·k·n/M)
```

Each coefficient is a circle. Its radius is `|c[k]|`, it starts at angle
`arg(c[k])`, and its arm turns `k` times per lap (negative `k` turns the
other way). Put the circles end to end, largest first, and the tip of the
last arm draws the border back:

```
z(t) ≈ Σ c[k] · e^(2πi·k·t)        t from 0 to 1
```

Keeping only the first N circles is a low-pass filter on the shape.
Five circles give a lumpy blob, ten and fifteen bring out the peninsula
and the north-east, one hundred follow the coastline, and two hundred
pick up the Rann of Kutch, the Sundarbans and the Jammu and Kashmir
border.

## Run it

```bash
pip install numpy matplotlib          # ffmpeg is needed only for .mp4 output

# side-by-side 5 / 10 / 15 / 100 / 200 circles  ->  fourier_india.gif
python fourier_india.py

# one panel with 100 circles, as a video
python fourier_india.py --circles 100 -o india_100.mp4

# any set of levels, longer and smoother
python fourier_india.py --circles 3 25 50 500 --frames 480 --fps 60 --dpi 150
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--circles`, `-n` | `5 10 15 100 200` | one panel per value |
| `--samples`, `-m` | `2048` | points the border is resampled to before the DFT (also the maximum number of circles) |
| `--frames` | `240` | frames for one full lap of the border |
| `--fps` | `30` | playback speed |
| `--hold` | `30` | frames to pause on the finished drawing |
| `--dpi` | `100` | output resolution |
| `--output`, `-o` | `fourier_india.gif` | `.gif` (Pillow) or `.mp4` / `.webm` (ffmpeg) |

`preview.gif` above was made from the `.mp4` with ffmpeg so it stays small:

```bash
ffmpeg -i fourier_india.mp4 -filter_complex \
  "[0:v] fps=20,scale=1400:-1:flags=lanczos,split [a][b];[a] palettegen=max_colors=64 [p];[b][p] paletteuse=dither=bayer:bayer_scale=5" \
  preview.gif
```

## Files

* `fourier_india.py` – resamples the outline, takes the FFT, animates the epicycles.
* `india_epicycles.html` – the same animation in the browser: open it, pick 5 / 10 / 15 / 100 / 200 or any number of circles with the slider, or compare all five side by side. No build step, no server.
* `india_outline.json` – 2401 lon/lat points of India's mainland border.
* `extract_india_outline.py` – regenerates `india_outline.json` from a Natural Earth GeoJSON file.

## Where the border comes from

The coordinates are the mainland polygon of the Natural Earth 1:10m
*Admin 0 – Countries (India point of view)* dataset, which draws India's
borders as India maps them. Natural Earth is public domain. The ring was
simplified with Douglas–Peucker (tolerance 0.01°) from 7497 to 2401
points; the Andaman, Nicobar and Lakshadweep islands are left out because
the epicycle chain can only draw one closed curve.

To rebuild the outline from a different Natural Earth file (for example
the default 1:50m countries file):

```bash
python extract_india_outline.py ne_10m_admin_0_countries_ind.geojson --tolerance 0.01
```

## How the script works

1. **Project** – longitude is scaled by `cos(mean latitude)` so the map is not stretched, then the curve is centred and scaled to unit size.
2. **Resample** – the border is re-spaced to `--samples` points equally spaced along its length, so densely digitised stretches of coast do not dominate the transform.
3. **Transform** – `numpy.fft.fft` gives the coefficients; they are sorted by radius and the `k = 0` term (a pure translation) is dropped.
4. **Animate** – for every frame the chain of the first N circles is evaluated at time `t`, and the path of the tip so far is drawn over a faint copy of the real border. All circles of a panel are drawn as one polyline separated by `NaN`s, which keeps 200 circles × 5 panels fast.
