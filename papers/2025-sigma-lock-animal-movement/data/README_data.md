This folder is used for input data for the paper
"Episodic σ-Locks and Topological Stability in Animal Movement".

Due to size and licensing constraints, the *raw tracking data* (GPS/ACC)
are **not** included in this repository. Instead, please download them from
the original sources:

- Straw-coloured fruit bats, white storks, griffon vultures, pigeons, etc.:
  see the "Data sources" section of the paper and the Movebank DOIs / study IDs.

Suggested workflow:

1. Obtain access to the corresponding Movebank studies.
2. Export the required tracks as CSV (one file per dataset), e.g.:

   - `storks_gps.csv`
   - `storks_acc.csv`
   - `griffons_positions.csv`
   - `bats_positions.csv`
   - etc.

3. Place these raw CSV files into `data/raw/` (not under version control).
4. Run the preprocessing scripts in `src/` to generate the smaller, derived
   CSV files that *are* tracked in Git, e.g.:

   - `window_states.csv`
   - `fragmentation_events.csv`
   - `event_profiles_summary.csv`
   - `triad_roles_per_stork.csv`
   - ...

Only the derived, relatively small CSVs should be committed to the repository.
