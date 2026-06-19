# Draft conventions

Paper: `DM.tex` (revtex4-2, two-column, APS prd). Compile from this folder:
```
pdflatex -interaction=nonstopmode DM.tex   # run twice
```

## Writing style

- **Reduce `;`, `:`, em-dashes (`---`), and en-dashes (`--`) in prose.** Prefer short period-separated sentences. If a colon feels natural, try splitting into two sentences first. Replace `---` parentheticals with commas or `( )`. Legitimate en-dashes stay: numeric ranges (`epochs 49--249`, `10--35`) and named compounds (`Metropolis--Hastings`, `Kullback--Leibler`).
- **Plain physics language over ML jargon.**
- **Let formulas speak when words are vaguer.**
- **Keep generic sections generic.**
- **Unpack ML-sounding claims operationally when they'd leave a physicist guessing.**

### APS-style scientific prose

Follow standard APS journal conventions:
- Direct, concise sentences in active voice ("We choose ...", "We adopt ...").
- Standard physics terminology in preference to ML-discipline jargon whenever an established physics equivalent exists.
- No marketing or pitch language ("physically interpretable", "principled", "expressive", "natural", "elegant").
- Assert findings directly; avoid hedging ("we believe", "in some sense", "it can be shown that").
- Define non-standard terms once. Do not redefine objects familiar to the target audience.

### Equations and their interpretation

- Provide physical interpretation that adds insight beyond what the formula itself displays.
- Merge near-identical equations that differ only in a parameter substitution.

### Compression

- Delete preambles such as "At this stage, ... has not yet been specified."
- Don't double-declare common framings ("throughout this work ..." said twice in close proximity).



## Notation

- **Time as subscript:** `\phi_0`, `\phi_t`, `\phi_{t_{\max}}`.
- **Terminal time is `t_{\max}`**, not `T` (reserved for temperature in later sections).
- **VE hyperparameter `\sigma`** in $g(t)=\sigma^t$ is \emph{not} the Song--Ermon $\sigma_{\max}$. They are related by $\sigma_{\max}=\sigma(1)$ via `\eq{eq:ve_sigma_schedule}`.


## Editing workflow

- Section-by-section, paragraph-by-paragraph under the user's direction. Don't pre-emptively rewrite sections the user hasn't asked about.
- When changing notation, use `Edit` with `replace_all: true` after confirming the pattern is unambiguous (e.g. `\phi(0)` → `\phi_0`).
- Before recommending a swap like `T` → `t_{\max}`, grep usages and flag collisions.
- After an edit, summarise the change in 2--4 bullets: what changed, why. Don't restate the prose.
