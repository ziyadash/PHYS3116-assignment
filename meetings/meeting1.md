# Meeting Minutes 1
## Agenda
| Date | 23 September 2026 |
|--|--|
| Attendees | Ziyad, Pranshu |
| Purpose | Go through project spec. get an initial feel of the data and load it into memory with pandas. |
| Next meeting | 30 September 2026 |
| Author | Ziyad |

## Statement of Contribution and Self-reflection
- Ziyad: Set up the codebase, wrote some comments, and uploaded CSV data to the repo. Also writing meeting minutes this week (but we'll take turns doing this). I didn't have a good idea of what the assignment was actually about until I actually took the time to do some reading today. I have much more clarity now, which is great, and I'm excited to get deeper into the project and analyse the data. 
- Pranshu: Helped with research and understanding one of the datasets. 

We both have strong programming skills because we study Computer Science, so we don't necessarily have to learn how to 'use Pandas.'

I wrote some code to load all the CSV data into Pandas Dataframes (DFs) and took a look at them. These are some notes on the data. 

### Harris Catalogue of Milky Way Globular Clusters

The Harris catalogue is the standard reference compilation of parameters for the Milky Way's globular clusters. In this dataset it is split across two CSVs, which I'll join on `ID`.

#### Identification and Coordinates

| Field  | Description | Units |
|--------|-------------|-------|
| `ID`   | Cluster identifier (e.g. NGC 104), used as the join key | — |
| `Name` | Common or alternative name (e.g. 47 Tuc), where one exists | — |
| `RA`   | Right Ascension (J2000) | hms |
| `DEC`  | Declination (J2000) | dms |
| `L`    | Galactic longitude | degrees |
| `B`    | Galactic latitude | degrees |

#### Distances and Positions

| Field   | Description | Units |
|---------|-------------|-------|
| `R_Sun` | Heliocentric distance (distance from the Sun) | kpc |
| `R_gc`  | Galactocentric distance (distance from the Galactic centre), assuming R₀ = 8.0 kpc | kpc |
| `X`     | Sun-centred Cartesian coordinate, positive towards the Galactic centre | kpc |
| `Y`     | Sun-centred Cartesian coordinate, positive in the direction of Galactic rotation | kpc |
| `Z`     | Sun-centred Cartesian coordinate, positive towards the North Galactic Pole | kpc |

#### Velocities

| Field     | Description | Units |
|-----------|-------------|-------|
| `v_r`     | Heliocentric radial velocity (line-of-sight velocity relative to the Sun) | km/s |
| `v_r_e`   | Uncertainty in `v_r` | km/s |
| `v_LSR`   | Radial velocity relative to the Local Standard of Rest | km/s |
| `sig_v`   | Central velocity dispersion (spread of stellar velocities in the core) | km/s |
| `sig_v_e` | Uncertainty in `sig_v` | km/s |

#### Structural Parameters

| Field    | Description | Units |
|----------|-------------|-------|
| `c`      | King-model central concentration, c = log(r_t / r_c), where r_t is the tidal radius. Core-collapsed clusters are assigned c = 2.5 | — |
| `r_c`    | Core radius | arcmin |
| `r_h`    | Half-light radius (radius containing half the cluster's total light) | arcmin |
| `mu_V`   | Central surface brightness in the V band | mag/arcsec² |
| `rho_0`  | Central luminosity density, given as log | log(L☉/pc³) |
| `lg_tc`  | Relaxation time at the core, given as log | log(years) |
| `lg_th`  | Relaxation time at the half-mass radius, given as log | log(years) |


### Krause Star Cluster Evolution Dataset
A dataset for studying star cluster formation and evolution. Our sample contains only globular clusters. Its main use is comparing clusters by mass, size, compactness, age and metallicity.

#### Identification

| Field     | Description | Units |
|-----------|-------------|-------|
| `Object`  | Main catalogue name of the cluster, usually from the NGC, IC or Messier (M) catalogues | — |
| `AltName` | Other names for the same cluster (e.g. M13 is also NGC 6205) | — |
| `Class`   | Cluster type. Every entry in this sample is `GC` (globular cluster), so this column carries no information here | — |

#### Physical Parameters

| Field   | Description | Units |
|---------|-------------|-------|
| `Mstar` | Total stellar mass of the cluster, given as log(M★/M☉) | log(M☉) |
| `rh`    | Half-mass (or half-light) radius, r_h: the radius that contains half the cluster's mass or light | pc |
| `C5`    | Compactness index, C₅ (defined below) | — |

#### Evolution and Chemistry

| Field  | Description | Units |
|--------|-------------|-------|
| `Age`  | Estimated age of the cluster | Gyr |
| `FeH`  | Metallicity [Fe/H]: the log ratio of iron to hydrogen, relative to the Sun. It traces how chemically enriched the gas was when the cluster formed. Negative values mean the cluster is metal-poor compared with the Sun | dex |

#### Compactness Index

C₅ is the cluster's mass in units of 10⁵ M☉, divided by its half-mass radius in parsecs:

$$
C_5 = \frac{M_\star / 10^5\,M_\odot}{r_h / \text{pc}}
$$

C₅ tracks how deep the cluster's gravitational potential well is. Clusters with C₅ > 1 are generally thought to be able to hold on to gas enriched by their first stars, so they can form multiple stellar populations. 

### VandenBerg Globular Cluster Ages Dataset

Ages and properties for 55 Galactic globular clusters, from VandenBerg et al. (2013), with structural and kinematic values merged in from other catalogues.

#### Identification

| Field     | Description | Units |
|-----------|-------------|-------|
| `dataset` | Source flag for the catalogue subset | — |
| `NGC`     | New General Catalogue number (e.g. 5272 for M3) | — |
| `Name`    | Common alternative name, e.g. Messier number | — |

#### Ages and Chemistry

| Field     | Description | Units |
|-----------|-------------|-------|
| `FeH`     | Metallicity [Fe/H]: log iron-to-hydrogen ratio relative to the Sun | dex |
| `Age`     | Absolute age, from fits to Victoria-Regina isochrones | Gyr |
| `Age_err` | Uncertainty in `Age` | Gyr |
| `Method`  | Age-dating technique, e.g. the ΔV(HB–TO) method (magnitude gap between the main-sequence turnoff and the horizontal branch) | — |
| `Figs`    | Figure numbers in the source paper showing the cluster's fit | — |
| `Range`   | Range of the colour–magnitude diagram used in the fit | — |
| `HBtype`  | Horizontal-branch morphology, (B − R)/(B + V + R). Runs from −1 (all red) to +1 (all blue) | — |

#### Position and Dynamics

| Field         | Description | Units |
|---------------|-------------|-------|
| `R_G`         | Galactocentric distance | kpc |
| `M_V`         | Integrated absolute V magnitude; more negative means brighter | mag |
| `v_e0`        | Central escape velocity, a measure of potential-well depth | km/s |
| `log_sigma_0` | log 10 central velocity dispersion | log(km/s) |


### The general plan for this assessment
I found this wonderful [paper](https://academic.oup.com/mnras/article/528/2/3198/7485911) that closely matches the objective of this assessment. We are analysing globular cluster datasets to determine whether any of the clusters are **ACCRETED** (born in smaller dwarf satellite galaxies that were later torn apart and absorbed by the Milky Way, bringing their clusters with them) or **IN-SITU** (born inside the main progenitor galaxy during its early gas-rich phase). There are probably differences to look out for in terms of orbit, metallicity, velocity, etc. and we need to analyse the data to see if any patterns emerge. 

Now we have a good feel of the data, and of our objectives. In the time leading up to our next meeting, we'll design methods to classify a GC as either in situ or accreted, based on the available data.

#### Action Items
**Data preparation**
- [ ] Standardise cluster names across the three datasets (e.g. I read that `NGC 104` and `47 Tuc` refer to the same cluster!), then merge Harris, Krause and VandenBerg into a single dataframe.
- [ ] Handle missing values and check units.

**Ground truth**
- [ ] Read the reference paper and see if it has any classifications of its own so we can use it as 'ground truth.'

**Exploratory analysis**
- [ ] Plot the age–metallicity relation (`Age` vs `FeH`) and check whether it splits into two branches. Do research into what to expect.
- [ ] Compare the spatial and kinematic distributions (`R_gc`, `Z`, `v_r`) of the labelled groups. Again, do research into what to expect.
- [ ] Compare structural properties (`rh`, `C5`, `c`, `HBtype`) between the labelled groups to see which features separate them.

**Method design**
- [ ] Propose at least one rule-based classifier (e.g. a dividing line in the age–metallicity plane) and at least one data-driven approach (e.g. clustering or logistic regression on the selected features).
- [ ] Define how we'll evaluate each method against the reference labels (e.g. accuracy, confusion matrix), with the small sample size in mind.

**Next meeting**
- [ ] Present the merged dataset, key plots and proposed methods, and agree on which method(s) to develop further.
