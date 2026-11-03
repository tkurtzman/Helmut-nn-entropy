# Split estimator

Multi-basin torsions are poorly described by a Gaussian, while bonds, angles and phase angles are close to harmonic, and a Gaussian captures all of their correlations at once. The split estimator (`--estimator split`, `estimate_split_entropy`) uses each where it fits. It divides the coordinates into a **soft** set S and a **hard** set H and writes

S = S<sub>soft</sub> + S<sub>hard</sub> + coupling

- **S<sub>soft</sub>**: the nearest-neighbor MIE over the soft coordinates (`--soft-estimator mie`, the default; `--mie-order` sets the order), the mutual information spanning tree (`mist`), MIST extended to several parents per coordinate (`network`; `--network-parents` sets how many, default 5; see [NN MIE](nn-mie.md)), or one joint nearest-neighbor estimate over all of them (`joint`, up to 8 soft coordinates). MIST uses the same single and pairwise nearest-neighbor entropies as MIE-2, but subtracts only the pairwise mutual information along the maximum spanning tree of the soft coordinates: one dependence per coordinate, the strongest that keeps the tree connected. With exact terms it is an upper bound on the joint entropy, while MIE-2 can fall far below it when many torsions are coupled.
- **S<sub>hard</sub>**: the quasi-harmonic entropy of the hard coordinates over all frames (`--hard-estimator qh`, the default), BQH (`bqh`), or QH truncated at first (`qh-mie1`) or second order (`qh-mie2`) in the mutual information expansion (see [below](#truncated-gaussian-hard-entropy)). Hard phase angles are centered on their circular mean (`--periodic-correlation centered`, the split default).
- **Coupling**: the dependence between the two sets, at most 0 up to sampling error. See [Coupling](#coupling).

The output lists each term before the total. Add ⟨ln J⟩ for a Cartesian entropy, as for every estimator.

## Which coordinates are soft

`--soft` (`soft=` in Python) chooses the soft set:

- **`topology`** (the default for files and for a Python `System`): the primary torsion about each rotatable bond, that is, about every bond not on a ring. With phase angles, the default, these are the coordinates of type `torsion`; methyl rotors count. Torsions about ring bonds, and every phase angle, bond and angle, are hard. This needs phase angles, because the raw secondary torsions of a rotor are as multimodal as the primary one.
- **`detect`** (the default for a raw array in Python): every periodic coordinate whose distribution has more than one basin, or is a free rotor (mean resultant length below 0.3), as found by [basin detection](#basin-detection).
- **a list** of coordinate numbers, counted from 1 on the command line (as printed by `--per-coordinate`) and from 0 in Python and Rust.

In Rust, `InternalCoordinates::soft_torsions()` gives the topological set; in Python, `System.soft_torsions`. `symmetric_rotor_torsions` lists the soft torsions about symmetric rotors (one end carries only two or more terminal atoms of one element, as in CH<sub>3</sub>, NH<sub>3</sub><sup>+</sup> and CF<sub>3</sub>), and `ring_torsions` those about ring bonds.

## Coupling

`--coupling` (`coupling=`) chooses how the dependence between the sets is accounted for:

- **`gaussian`** (the default): the Gaussian mutual information between the sets, `½ [ln|C(S∪H)| − ln|C(S)| − ln|C(H)|]`. Soft torsions enter the correlation matrix as cos kφ and sin kφ for k = 1 to 3 (`--soft-features harmonics`, the default; `--harmonics N` sets the number), as cos φ and sin φ only (`cossin`, the same as `--harmonics 1`), or as `e^{iφ}` (`complex`). As separate real columns, cos φ and sin φ can correlate with a hard coordinate separately: a bond angle that responds to trans against gauche but not to which gauche correlates with cos φ alone, which the single complex entry partly averages away. Higher harmonics catch dependences that are not linear in cos φ and sin φ, such as an angle that follows cos 3φ, strained at every eclipsed position. Each harmonic adds two columns per soft torsion, and the frames must outnumber all the columns.
- **`states`**: the hard entropy is computed within each conformational state of the soft torsions, the states' entropies are averaged with their populations, and the coupling is the difference from the hard entropy over all frames. A state is the combination of basins of the state-defining torsions: by default (`--state-torsions auto`) the soft torsions with more than one basin, excluding symmetric rotors, whose basins are equivalent; or a list. States with fewer than `--min-state-frames` frames (300 by default) or than the number of hard coordinates plus 2 are dropped and the others' populations renormalized; the output lists every state and the dropped frames, and warns when more than 5% are dropped. In Python, `state_labels` gives each frame's state directly.
- **`nn`**: the nearest-neighbor pairwise mutual information between the sets, `Σ I₂(xᵢ, xⱼ)` over every i in the hard set and j in the soft set, subtracted. These are the hard-soft cross terms of the second-order expansion, estimated exactly as the soft set's own pair terms are, with the periodic metric wherever the coordinate is periodic, and shared equally between the two coordinates of each pair under `--per-coordinate`. It costs one two-dimensional nearest-neighbor search per hard-soft pair, so the work grows as the product of the two set sizes. Use it to measure what `none` drops: on butane, where no physical hard-soft coupling exists, the cross terms were dominated by estimator noise and differed by more than an order of magnitude between two independent datasets of the same system.
- **`none`**: the sets are treated as independent. The cross terms are dropped rather than estimated; `nn` estimates them.

No mixing term is added: the entropy of moving between states is already part of S<sub>soft</sub>.

## Basin detection

`detect_basins` (Rust `basins::detect_basins`, Python `molentropy.detect_basins`) finds the basins of a periodic coordinate:

1. Histogram the values over one period (72 bins, fewer for small samples) and smooth with a wrapped Gaussian (standard deviation 2 bins).
2. Take every circular local maximum as a peak.
3. Merge neighboring peaks whose separating minimum is at least half the lower peak.
4. Put the basin boundaries at the remaining minima.
5. Merge basins holding less than 2% of the frames into a neighbor.

The result is deterministic and does not depend on where the period starts.

## Truncated Gaussian hard entropy

`--hard-estimator qh-mie1` (`hard_estimator="qh-mie1"`, `HardEstimator::QuasiHarmonicMie1`) keeps only QH's first-order term, the Gaussian entropy of each hard coordinate, Σ<sub>i</sub> ½ ln(2πe σ<sub>i</sub>²): the hard coordinates as independent Gaussians, with no correlation term. It is an upper bound on the QH hard entropy, since ½ ln|C| ≤ 0. It shows how much the hard correlations contribute; it is not a more accurate hard entropy. On alanine dipeptide ½ ln|C| was −1.68 nats at λ = 0 and −2.04 at λ = 1, so dropping it lowers S0 − S1 by 0.36 nats.

`--hard-estimator qh-mie2` (`hard_estimator="qh-mie2"`, `HardEstimator::QuasiHarmonicMie2`) keeps QH's Gaussian entropy of each hard coordinate but replaces its correlation term ½ ln|C| with the second-order mutual information expansion, the Gaussian mutual information of each pair subtracted once:

S<sub>hard</sub> = Σ<sub>i</sub> ½ ln(2πe σ<sub>i</sub>²) + ½ Σ<sub>i&lt;j</sub> ln(1 − ρ<sub>ij</sub>²)

where ρ<sub>ij</sub> is the correlation of hard coordinates i and j (its modulus with `--periodic-correlation complex`). This is the MIE-2 of the soft set with Gaussian terms. For Gaussian coordinates the full ½ ln|C| is exact, and the pairwise sum is an approximation to it: it agrees when the dependences are weak or confined to separate pairs, and misses the dependence shared among three or more coordinates. That error has either sign: the pairwise sum overcounts redundant dependences (three coordinates all correlated with one another) and undercounts synergistic ones (two independent coordinates that together determine a third). On butane the pairwise term was 0.03–0.06 nats more negative than ½ ln|C|; on alanine dipeptide (ff14SB, 41 hard coordinates) it was 0.01–0.02 less negative, and S0 − S1 moved by 0.005 nats. Use it to compare with the MIE-2 treatment of the soft set, not as a more accurate hard entropy. The coupling term is unchanged: it still uses ln|C(H)| of the whole hard set. With `--per-coordinate`, each pair's term is shared equally between its two coordinates; with `qh-mie1` each hard coordinate takes its first-order term.

## Choosing settings

The defaults (topological soft set, MIE-2 soft, centered QH hard, Gaussian coupling with three harmonics) were chosen on 18 small molecules and 4 peptide systems with MBAR references, and checked on 8 more small molecules and a held-out folded peptide (Trp-cage); see [Validation](validation.md#small-molecules-and-peptides-at-300-k), which also lists the best estimator for each system.

### Why these defaults

Each default is the setting with the lowest error on small flexible molecules that also works on every system. A simpler setting was preferred when errors were within about 0.05 nats. The numbers are RMSEs in nats over the 15 molecules used to choose the defaults, and over the 8 held out.

- **Topological soft set**, not detection. Detection was a little better on the first 15 molecules (0.59 against 0.70 with cos/sin coupling) but worse on the held-out 8 (0.66 against 0.61). It also makes ring torsions soft, which put a puckering cyclohexane 8–10 nats off with MIE-2. Topology is deterministic, needs no sampling to classify a torsion, and treats a rotatable bond the same whether or not it crossed a barrier in the trajectory.
- **MIE-2 soft entropy**, not MIST or the joint estimate. MIE-2 had an RMSE of 0.38 on the 23 small molecules, against 0.61 for MIST. The joint estimate was no better (0.69 on the 13 molecules where it ran, against 0.70), its bias grows with the number of soft torsions, and it stops at 8, so it failed on decane and longer. MIST is an option because it is far better on folded peptides, where MIE-2 overcounts the dependence between many coupled torsions; see below.
- **QH for the hard set**, with phase angles centered on their circular mean. BQH hard entropies were within 0.02 nats of QH on the first 15 molecules and worse on the held-out 8 (0.71 against 0.61). QH is also faster, and the hard coordinates are close to Gaussian by construction. On butane, centered phase angles beat complex ones by 0.02 nats; on the 15 molecules the difference was 0.01.
- **Gaussian coupling**, not per-state. No coupling at all left S0 − S1 too low (1.16 and 1.01). Per-state coupling was the most accurate on the first 15 molecules where it ran (0.33), but it fails when no state has enough frames (decane, hexadecane, both peptides in GBn2), and it was worse on the held-out 8 (0.66). Gaussian coupling always runs.
- **Three harmonics** of each soft torsion in the coupling. The error fell with each harmonic: 0.70, 0.55 and 0.36 for one, two and three on the first 15 molecules, and 0.61, 0.54 and 0.41 on the held-out 8. One-hot basin indicators added to cos/sin did about as well as two harmonics (0.55 and 0.56). More harmonics were not tested; each adds two correlation columns per soft torsion and needs more frames. Three harmonics are slightly worse than cos/sin on butane and hexane (by 0.05–0.09 nats), where cos/sin was already within 0.04.

- **Small flexible molecules** (up to about 15 soft torsions): on 23 such molecules the defaults were within 0.5 nats on 20 and 0.84 at worst, and beat both QH and BQH on all but hexane, where BQH was 0.06 nats closer. (With a benzamide's ring torsions switched off as well, QH was better.) With one multi-basin torsion, as in butane, plain BQH is about as good.
- **Per-state coupling** (`--coupling states`) is more accurate when there are a few well-sampled conformers: about 4 or fewer state-defining torsions and plenty of frames in each state. It cut the error from 0.31 to 0.02 nats on biphenyl and from 1.6 to 0.3 on a nitrobenzamide. With more states it drops frames or fails, and on hexane it was 0.5 nats worse than Gaussian coupling.
- **Peptides and other large molecules**: use `--soft-estimator network`. It was stable across strides and roughly halved MIST's error on every peptide tested, to 4–12 nats in S0 − S1 (1.5–5% of the entropy change), and to 4 nats on the solvation entropy of the GS chain ([Validation](validation.md#bayesian-network-soft-estimator)). With BQH for the hard set (`--hard-estimator bqh`) it was 5.6 nats off on Trp-cage. No estimator was reliable to a few nats on peptides, and all of them overestimated the entropy of the more structured state. With the MIE-2 default, the split estimator overestimated the entropy loss of a folded β-hairpin (chignolin) by 30–165 nats, because MIE-2 subtracts every pairwise dependence among its 50 soft torsions, and on Trp-cage (99 soft torsions) its error jumped by up to 10 nats between strides. MIST was within 3.6 nats on chignolin in implicit solvent but 17 nats low on Trp-cage; BQH was 5.3 and 25 off. Check any result on every 2nd and 4th frame ([Finite-sampling bias](finite-sampling-bias.md#checking-a-result)). Above 20 soft coordinates the CLI and Python warn (`LARGE_SOFT_SET` in Rust).
- **Compact small molecules**: a zwitterion collapsed into a salt bridge was off by 1.0 nats with only 4 soft torsions.
- **Flexible rings**: torsions about ring bonds are hard under the topological rule, so a puckering ring falls to the Gaussian; on cyclohexane this was within 0.6 nats. Do not combine `--soft detect` with MIE-2 there: detection makes the ring torsions soft, and the pairwise MIE cannot represent the ring closure that couples them (8–10 nats off). The joint soft estimator handled it (0.14 nats).
- **The joint soft estimator** has no truncation error but its bias grows with the number of soft torsions, and it is limited to 8.
- **Detection** can call a rotatable bond stuck in one well hard, and a stiff torsion that samples two wells soft; the topological rule cannot.

`--per-coordinate` (`per_coordinate=True` in Python, `estimate_coordinate_split_entropy` in Rust) gives each coordinate's share of the soft or hard entropy and of the coupling, except with the joint soft estimator; see [Per-coordinate entropies](per-coordinate.md#the-split-estimator). `--mutual-information` does not apply to the split estimator.
