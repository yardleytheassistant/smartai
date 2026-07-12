# PeptideTrack

A private, offline iPhone app for tracking research peptides in a study —
reconstitution math, dose calculations for insulin syringes, an injection
log, and supply inventory (bacteriostatic water, syringes, safety materials).

It's a Progressive Web App: pure HTML/CSS/JavaScript with **no backend and no
accounts**. All data is stored in your browser's local storage on your own
device. Nothing is uploaded anywhere.

## What it does

- **Vials** – Record each peptide vial (mg per vial, mL of bacteriostatic
  water added). It computes the concentration and **mcg per unit** on the
  syringe automatically.
- **Dose calculator** – Pick a vial + target dose (mcg) and it tells you how
  many **units to draw** on a U-100 insulin syringe (and the mL). Includes a
  standalone quick converter and a warning if a dose exceeds a 1 mL syringe.
- **Injection log** – Log each injection with dose, auto-suggested rotating
  site, date/time, and notes. Exports to CSV. Auto-deducts a syringe and prep
  pad from supplies.
- **Supplies** – Track bacteriostatic water, insulin syringes, alcohol prep
  pads, sharps container, gauze, etc., with low-stock alerts on the dashboard.
- **Backup** – Export/import all data as a JSON file (Settings ⚙️).

## Install on your iPhone

1. Host the `peptide-tracker/` folder somewhere your phone can reach over
   HTTPS. Easiest option: **GitHub Pages**.
   - In the repo Settings → Pages, serve from this branch.
   - Open `https://<your-user>.github.io/<repo>/peptide-tracker/` in **Safari**.
2. Tap the **Share** button → **Add to Home Screen**.
3. Launch it from the home-screen icon — it runs full-screen and works
   offline.

To try it locally on a computer:

```bash
cd peptide-tracker
python3 -m http.server 8000
# open http://localhost:8000
```

(A local `file://` open works for most features, but the service worker /
offline caching needs `http(s)://`.)

## Dosing math

For a U-100 insulin syringe, 100 units = 1 mL.

```
concentration (mg/mL) = vial_mg / bac_water_mL
mcg per unit          = vial_mg * 10 / bac_water_mL
units to draw         = dose_mcg * bac_water_mL / (10 * vial_mg)
```

Example: a 5 mg vial + 2 mL bacteriostatic water = 2.5 mg/mL = 25 mcg per
unit. A 250 mcg dose = 10 units = 0.1 mL.

## Disclaimer

This tool is for **research record-keeping and dose arithmetic only**. It is
not medical advice and does not endorse or recommend administering any
substance. Verify every calculation yourself, follow your study protocol and
all applicable laws, handle materials safely, and dispose of sharps properly.
Because data is stored only in your browser, clearing Safari's website data
will erase it — export a JSON backup regularly.
