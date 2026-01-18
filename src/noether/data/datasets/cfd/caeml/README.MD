# DrivAerML CFD Dataset

This dataset contains CFD simulation data for car models from the DrivAerML dataset. The dataset includes surface and volume properties for 500 different car geometries.

## Dataset Overview

- **Total Samples**: 500 simulations
- **Training Set**: 400 samples
- **Test Set**: 50 samples
- **Validation Set**: 34 samples
- **Hidden Test Sets**: 16 samples
- **Data Types**: Surface pressure, surface friction (i.e., wallshear stress), volume velocity, volume vorticity, and volume pressure
## Quick Start

### 1. Download Raw Data


### 2. Preprocess the Data

#### Validate Preprocessing (Dry Run)


#### Run Full Preprocessing


#### Advanced Preprocessing Options

### 3. Verify Preprocessing

## Data Format

Each preprocessed simulation contains the following files:

| File | Shape | Description |
|------|-------|-------------|
| `surface_points.pt` | `(N_surf, 3)` | 3D coordinates of surface points |
| `surface_pressure.pt` | `(N_surf,)` | Pressure values at surface points |
| `surface_normals.pt` | `(N_surf, 3)` | Unit normal vectors at surface points |
| `volume_points.pt` | `(N_vol, 3)` | 3D coordinates of volume points |
| `volume_velocity.pt` | `(N_vol, 3)` | Velocity vectors at volume points |
| `volume_normals.pt` | `(N_vol, 3)` | Unit normal vectors at volume points |
| `volume_sdf.pt` | `(N_vol,)` | Signed distance field w.r.t. car body |

## Computing the statistics of the dataset

```bash 
python dataset_statistics.py --dataset_kind=emmi.datasets.cfd.DrivAerMLDataset" --root=/path/to/output/preprocessed" --split=train --exclude_attributes=surface_normals,volume_normals,volume_sdf --log_scale=volume_vorticity"
```

## Dataset Source

Original dataset: 




surface_normal_stl.pt                
surface_position_stl.pt                
surface_position_vtp.pt  
surface_wallshearstress.pt  
volume_cell_pressure.pt                  
volume_cell_totalpcoeff.pt  
volume_cell_vorticity.pt
surface_normal_stl_resampled100k.pt  
surface_position_stl_resampled100k.pt  surface_pressure.pt      volume_cell_position.pt     volume_cell_surface_distance_cKDtree.pt  volume_cell_velocity.pt