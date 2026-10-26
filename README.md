# DHAM Skin Permeability

Estimate the **skin permeation coefficient (kP)** of a small molecule from GROMACS umbrella sampling simulations of that molecule crossing a model stratum corneum membrane.

The free energy profile (PMF) comes from the **Dynamic Histogram Analysis Method (DHAM)**. Position-dependent diffusion comes from a **Kramers–Moyal expansion** of the DHAM Markov model. The two are combined in the **inhomogeneous solubility–diffusion model** to give kP. You can also add a shunt-route term (sweat glands and hair follicles).

---

## Contents

- [Method](#method)
- [Requirements](#requirements)
- [Usage](#usage)
- [Input files](#input-files)
- [Output](#output)
- [Shunt-route model](#shunt-route-model)
- [Conventions and assumptions](#conventions-and-assumptions)
- [References](#references)

---

## Method

The pipeline is driven by `dham()` in `src/dham/dham.py`:

1. **Read the simulation data.** Reads the umbrella centres, force constants and temperature from the GROMACS `.log` files, and the reaction coordinate from the `pullx.xvg` files.
2. **Drop equilibration.** `autocut` finds and removes the equilibration portion at the start of each trajectory.
3. **Tune the Markov model.** `mm_tuning` picks the number of bins and the lagtime from the implied timescales.
4. **Calculate the free energy profile.** Builds an unbiased Markov model with DHAM and takes its stationary distribution as the PMF.
5. **Calculate diffusion.** `km_diffusion` estimates D(z) from the Kramers–Moyal expansion of the Markov model.
6. **Calculate permeability.** `calc_perm` integrates the PMF and D(z) with the inhomogeneous solubility–diffusion model to give kP for the stratum corneum.
7. **Add the shunt routes (optional).** If you give the molecule's diffusion coefficient in water, it estimates kP for the two shunt routes, **sweat glands** and **hair follicles**, using a modified version of the Mitragotri (2003) model. It adds both to the stratum corneum kP. Each route corrects the aqueous diffusion coefficient for the viscosity of the fluid filling it. See [Shunt-route model](#shunt-route-model).
8. **Estimate uncertainties.** Splits the data into four consecutive blocks and repeats the analysis on each. The error on the PMF, D(z) and kP is the standard deviation across the blocks.

---

## Requirements

- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numba](https://numba.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

```bash
pip install numpy scipy matplotlib numba tqdm
```

---

## Usage

```bash
python src/dham/dham.py -il log.dat -ix px.dat [-watdiff DIFFUSION UNCERTAINTY]
```

| Argument   | Required | Description |
|------------|----------|-------------|
| `-il`      | Yes      | Text file listing the GROMACS umbrella sampling `.log` files, one per line, in window order |
| `-ix`      | Yes      | Text file listing the matching `pullx.xvg` files, one per line, in the **same order** |
| `-watdiff` | No       | Diffusion coefficient of the molecule in water and its uncertainty, in cm² s⁻¹ (e.g. `1E-5 1E-6`). Turns on the shunt-route correction. |

---

## Input files

**`.log` files** (one per umbrella window). These must contain:
- `ensemble-temperature`
- `init`: the umbrella centre, in nm
- `k`: the bias force constant, in kJ mol⁻¹ nm⁻²

If a log has more than one pull coordinate, the last one is used.

**`pullx.xvg` files** (one per umbrella window):
- The first 17 lines are skipped as header.
- Column 1 is time (ps) and column 2 is the pull coordinate (nm).
- All windows must have the **same number of frames** and the **same output interval**.

---

## Output

Results go to `./DHAM_Results/`, which is created in the directory you run from.

| File | Contents |
|------|----------|
| `kp.txt` | kP for the stratum corneum, in cm s⁻¹, with log₁₀ values. With `-watdiff`, it also gives the combined shunt kP (sweat glands + hair follicles) and the total kP. |
| `PMF.png`, `PMF.xvg` | Free energy profile ΔG(z) in kJ mol⁻¹, with error |
| `Smoothed_Diffusion.png`, `Diffusion.xvg` | Diffusion profile D(z) in cm² s⁻¹, with error |
| `Histograms.png` | Combined umbrella sampling histogram, to check overlap and sampling |
| `Convergence.png` | Convergence of the PMF |
| `Discretization.png`, `Discretization_Optimization.xvg` | Implied timescales against bin number |
| `Lagtime_Optimization.png`, `Lagtime_Optimization.xvg` | Implied timescales against lagtime |

Example `kp.txt` (with `-watdiff 5.94E-5 9E-7`):

```
kP_StCor (cm/s) = 1.80e-09 +/- 6.26e-10
kP_Shunt (cm/s) = 1.27e-07 +/- 1.92e-09
kP_Total (cm/s) = 1.29e-07 +/- 2.55e-09
```

---

## Shunt-route model

The shunt routes bypass the stratum corneum. They are based on the Mitragotri (2003) pore model, kP = ε · D / L, applied separately to sweat glands and hair follicles. In each route, the molecule's aqueous diffusion coefficient is scaled by the ratio of the viscosity of water to the viscosity of the fluid filling that route:

$$
k_\text{P}^\text{Shunt} = k_\text{P}^\text{Sweat} + k_\text{P}^\text{Hair}
$$
$$
k_\text{P}^\text{Sweat} = \frac{\epsilon_\text{Sweat} D_\text{Water} V_\text{Water}}{L_\text{Sweat} V_\text{Sweat}}
$$
$$
k_\text{P}^\text{Hair} = \frac{\epsilon_\text{Hair} D_\text{Water} V_\text{Water}}{L_\text{Hair} V_\text{Hair}}
$$

| Parameter | Symbol | Sweat glands | Hair follicles |
|-----------|--------|--------------|----------------|
| Fraction of skin area | ε | 1 × 10⁻⁴ (Mitragotri, 2003) | 2 × 10⁻³ (Mitragotri, 2003) |
| Path length | L | 500 µm (Mitragotri, 2003) | 410 µm, follicular infundibulum (Kasting *et al.*, 2019) |
| Filling fluid | | Sweat | Sebum |
| Viscosity of the filling fluid | V | 0.92 mPa s (Emrich *et al.*, 1968) | 85.97 mPa s (Butcher & Coonin, 1949) |
| Viscosity of water at 303.15 K | V_water | 0.78 mPa s (Coe & Godfrey, 1944) | 0.78 mPa s (Coe & Godfrey, 1944) |

D_water is the value given with `-watdiff`. The same formula is applied to the `-watdiff` uncertainty to give the shunt error.

The stratum corneum and shunt routes act in parallel, so their permeabilities add:

$$
k_\text{P}^\text{Total} = k_\text{P}^\text{Lipids} + k_\text{P}^\text{Shunt} 
$$

---

## Conventions and assumptions

- **The membrane is assumed symmetric about z = 0.** The code uses |z| and mirrors the final profiles to cover −z to +z.
- **Units.** Distances are in Å internally (nm in the inputs). Bias constants and Markov model energies are in kT. The reported PMF is in kJ mol⁻¹.
- **Valid range.** The PMF and D(z) are only reported between the smallest and largest umbrella centres.
- **Smoothing.** The PMF in `PMF.png` is lightly smoothed. `PMF.xvg` and the kP calculation use the unsmoothed PMF.
- **Lagtimes** are in trajectory frames, so multiply by the `pullx` output interval to get a time.

---

## References

- Rosta, E. & Hummer, G. (2015). Free energies from dynamic weighted histogram analysis using unbiased Markov state model. *J. Chem. Theory Comput.*, 11(1), 276–285. https://doi.org/10.1021/ct500719p
- Mitragotri, S. (2003). Modeling skin permeability to hydrophilic and hydrophobic solutes based on four permeation pathways. *J. Control. Release*, 86(1), 69–92. https://doi.org/10.1016/s0168-3659(02)00321-8
- Emrich, H. *et al.* (1968). Sweat composition in relation to rate of sweating in patients with cystic fibrosis of the pancreas. *Pediatr. Res.*, 2, 464–478. https://doi.org/10.1203/00006450-196811000-00004
- Butcher, E. O. & Coonin, A. (1949). The physical properties of human sebum. *J. Invest. Dermatol.*, 12(4), 249–254. https://doi.org/10.1038/jid.1949.37
- Coe, J. & Godfrey, T. (1944). Viscosity of water. *J. Appl. Phys.*, 15, 625–626. https://doi.org/10.1063/1.1707481
- Kasting, G. B., Miller, M. A., LaCount, T. D. & Jaworska, J. (2019). A composite model for the transport of hydrophilic and lipophilic compounds across the skin: steady-state behavior. *J. Pharm. Sci.*, 108(1), 337–349.
