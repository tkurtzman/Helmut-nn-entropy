# Coordinates

By default, file-based calculations use every bond between two heavy atoms (ring closures included), plus the BAT tree's angles and torsions, with phase angles in place of redundant torsions. Bonds to hydrogen are left out because they are usually constrained.

**Tree bonds** (`--tree-bonds`, `tree_bonds=True`, `BondSet::Tree`) use the bonds of the BAT spanning tree: the two root bonds and the bond that places each later atom, with hydrogens included and ring closures left out. Together with the angles and torsions, this gives exactly 3N - 6 non-redundant coordinates per molecule (one bond for a diatomic, including one with hydrogen). This is the coordinate set the BAT Jacobian is defined for. With constrained bonds to hydrogen, keep the default; see [Choosing the bond set](#choosing-the-bond-set).

**Tree bonds without the constrained ones** (`--bonds tree-unconstrained`, `bonds='tree-unconstrained'`, `BondSet::TreeUnconstrained`) build the same spanning tree and so keep exactly the same angles and torsions, but drop the bonds to hydrogen, which SHAKE and the other constraint algorithms hold at a fixed length. The set is still non-redundant, and the Jacobian is defined for it. This is the right choice for a trajectory with constrained X-H bonds; see [Choosing the bond set](#choosing-the-bond-set).

**Phase angles** (on by default; `--no-phase-angles`, `phase_angles=False` or `CoordinateOptions::phase_angles` turn them off) group the torsions by central bond. The first torsion about each bond, in coordinate order, keeps its value. Every later torsion about that bond is replaced by its offset from the first one, wrapped to (-π, π]. The offset is still a 2π-periodic coordinate. Because the map is a shear with unit Jacobian, the joint entropy of the full coordinate set is unchanged. A truncated MIE is not invariant, though. The raw torsions of atoms that rotate together, such as the three hydrogens of a methyl group, are strongly correlated, and a second-order expansion misses part of that correlation. Their offsets are stiff and nearly independent of the reference torsion, so less correlation is left beyond second order. Every estimator in the crate does better with them; see [Validation](validation.md). Phase angles work with either bond set and with `--torsions-only`.

**The BAT Jacobian** (`--jacobian`, `System.mean_log_jacobian`, `InternalCoordinates::log_jacobian`) is ln J = Σ 2 ln b + Σ ln sin θ over the tree's bonds and angles. Adding its mean to the BAT entropy gives the entropy of the internal motion measured in Cartesian coordinates. Bond lengths are in the trajectory's units (Å for Amber), so ln J depends on that choice, just as the bond-length entropies do. Phase angles do not change it.

## Choosing the bond set

Keep the default heavy-atom bonds for trajectories with constrained bonds to hydrogen (SHAKE, RATTLE, LINCS), which most production simulations use, or `--bonds tree-unconstrained` when the Jacobian is wanted as well. Use tree bonds only for unconstrained simulations.

A constrained bond does not fluctuate, so as a coordinate it contributes only the noise of the constraint solver and of the stored precision. That noise differs between ensembles, so it does not cancel in a difference. Tree bonds add one such coordinate per hydrogen. On the 23 small molecules and alanine dipeptide of the [validation](validation.md#small-molecules-and-peptides-at-300-k), all run with X-H bonds constrained, tree bonds made every estimator worse. The table gives the RMS error in S0 − S1 against the HREX/MBAR references (nats, means of three replicates, the Jacobian included in both):

| estimator | heavy-atom bonds | tree bonds | systems where tree bonds did better |
|---|---|---|---|
| split (default) | **0.37** | 2.00 | 7 of 23 |
| split, MIST | **0.69** | 2.14 | 8 of 23 |
| BQH | **1.78** | 3.35 | 9 of 23 |

The largest changes were in the alkanes, whose hydrogens make up most of the extra coordinates. For hexane, the split estimator's error went from +0.12 to −7.7 nats and BQH's from +0.06 to +10.7. For butane, the split estimator's went from +0.08 to −3.8. Without butane and hexane, the split estimator's RMS error was still 0.39 with heavy-atom bonds against 0.92 with tree bonds.

The Jacobian is defined only for a tree set, so `--jacobian` and `mean_log_jacobian` need `--bonds tree` or `--bonds tree-unconstrained`. With constraints, use `tree-unconstrained`: it gives the entropy of the unconstrained coordinates and the Jacobian in one pass, and leaves out the constrained bonds' 2 ln b terms, each of which is constant and cancels in a difference between ensembles. For a molecule with no rings the coordinate set is the same as the default heavy-atom one, so the entropy is unchanged; with rings it drops the ring-closure bonds the heavy-atom set keeps, which is what makes it non-redundant.

## Rotatable, ring and rotor torsions

Each torsion is also classified by the bond it turns about, for the [split estimator](split.md):

- **Soft torsions** (`soft_torsions`): the primary torsion about each bond that is not on a ring. With phase angles these are the coordinates of type `torsion` off rings.
- **Symmetric rotor torsions** (`symmetric_rotor_torsions`): soft torsions about a bond one end of which carries only two or more terminal atoms of one element, as in CH<sub>3</sub>, NH<sub>3</sub><sup>+</sup> and CF<sub>3</sub>. Their basins are equivalent.
- **Ring torsions** (`ring_torsions`): torsions about a bond that lies on a ring, found as the bonds that are not bridges of the molecule's bond graph.

In Rust these are methods of `InternalCoordinates`; in Python, properties of a `System`. They depend only on the topology, and `--torsions-only` keeps them.

## Which atoms are used

- File-based calculations select the solute automatically. Water is excluded when its atom type is an Amber water type (`OW`, `HW`, `EP`, `EPW`) or its residue is a water name (`WAT`, `HOH`, `SOL`, `TIP3`, `SPCE`, `OPC`, and similar), so four-point models and non-Amber water types are handled. Massless virtual sites are excluded because they carry no degrees of freedom. Monatomic species such as counter-ions contribute nothing, since they have no internal coordinates, and a diatomic contributes only its bond length (with the default bond set, only when both atoms are heavy). Solute atoms may appear anywhere in the topology.
- The trajectory must have exactly as many atoms as the topology. A stripped topology paired with a solvated trajectory is rejected rather than read on the assumption that atoms line up.
