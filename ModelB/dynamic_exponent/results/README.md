# Result sets

- `preliminary_modes_L*.jld2`: primary `dt=0.005` trajectories for `L=8–64`.
- `preliminary_fit/`: primary literature-matched fit.
- `preliminary_dt0p0025_modes_L*.jld2` and `preliminary_fit_dt0p0025/`:
  half-step check on `L=8–32`.
- `preliminary_fit_dt0p005_L8_L32/`: matching three-size baseline for the
  half-step comparison.
- `preliminary_fit_free_amplitude/`: systematic check using
  `A exp(-t/tau)`.
- `preliminary_fit_window_0p7_0p15/`: fit-window systematic check.

These are preliminary results, not the final quoted exponent. See
`../preliminary_results.md` for interpretation and remaining systematics.
