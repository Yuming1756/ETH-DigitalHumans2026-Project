# DigitalHumans Project

This repository contains the project-specific scripts for comparing **TokenHMR** and **4D-Humans** reconstruction results with **EgoHumans** ground-truth meshes.

The pipeline includes:

1. Preparing prediction outputs from TokenHMR or 4D-Humans.
2. Extracting EgoHumans groundtruth meshes, ego-device targets locations and 2D keypoints.
3. Running optimization.
4. Visualizing baseline and optimized meshes.
5. Computing evaluation metrics against EgoHumans ground truth.

The results of the first and second step were uploaded to google drive. You can refer to the 2. Download the Data session.

Large data files, SMPL model files, prediction outputs, optimized meshes, and generated `.npz` / `.obj` files are **not included** in this GitHub repository. They should be placed locally according to the folder structure below.

---

## Folder structure

```text
ETH-DigitalHumans2026-Project/
├── README.md
├── .gitignore
│
├── smpl/
│   └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
│
├── data/
│   ├── mesh_cam_unscaled/
│   │   └── cam01/
│   │       └── rgb/
│   │           ├── 00001/
│   │           │   ├── mesh_aria01.obj
│   │           │   ├── mesh_aria02.obj
│   │           │   ├── mesh_aria03.obj
│   │           │   └── mesh_aria04.obj
│   │           ├── 00002/
│   │           │   ├── mesh_aria01.obj
│   │           │   ├── mesh_aria02.obj
│   │           │   ├── mesh_aria03.obj
│   │           │   └── mesh_aria04.obj
│   │           └── ...
│   │
│   └── poses2d/
│       ├── aria01/
│       │   ├── left/
│       │   ├── rgb/
│       │   └── right/
│       ├── aria02/
│       ├── aria03/
│       ├── aria04/
│       └── cam01/
│           └── rgb/
│               ├── 00001.npy
│               ├── 00002.npy
│               └── ...
│
├── head_targets_unscaled/
│   ├── 00001_aria01_cam01_egohumans_style.txt
│   ├── 00001_aria02_cam01_egohumans_style.txt
│   ├── 00001_aria03_cam01_egohumans_style.txt
│   └── 00001_aria04_cam01_egohumans_style.txt
│
├── src/
│
├── TokenHMR/
│   └── demo_out/
│       ├── my_image_with_smpl_params
│
├── 4D-Humans/
│   └── demo_out/
│       ├── my_image_with_smpl_params/
│
├── extract_all_aria_targets.sh
├── run_baseline_all.sh
├── run_smplify_v1_frame.sh
├── run_smplify_v2_frame.sh
├── evaluate_frames_00001_00006.sh
├── visualize_baseline_vs_egohumans.sh
├── visualize_smplify_frame.sh
├── visualize_smplify_v2_frame.sh
└── run_v1_v2_and_ablation.sh
```

## 2. Download the Data

The code is hosted on GitHub, while the data and generated prediction files are hosted on Google Drive.

Download the following files from Google Drive:

```text
release_data.tar.gz : https://drive.google.com/file/d/1Y1mA5CciYtqlq34S3D-ieqp-J_iIgnnZ/view?usp=sharing
release_data.tar.gz.sha256 : https://drive.google.com/file/d/1Re9haqnPJORknOzDWyBgy2b_t4jYhVmg/view?usp=sharing
```

Place them inside the cloned `ETH-DigitalHumans2026-Project/` folder:

```bash
cd ./ETH-DigitalHumans2026-Project
```

Verify the checksum:

```bash
sha256sum -c release_data.tar.gz.sha256
```

Unpack the archive:

```bash
tar -xzvf release_data.tar.gz
```

Copy the released data into the project root:

```bash
rsync -avh release_data/ ./
```

After this step, verify that the required folders exist:

```bash
ls data/mesh_cam_unscaled/cam01/rgb
ls data/poses2d/cam01/rgb
ls keypoints2d
ls head_targets_unscaled
ls TokenHMR/demo_out
ls 4D-Humans/demo_out
```

---

## 3. SMPL Model

Please download the neutral SMPL model separately and place it here:

```text
ETH-DigitalHumans2026-Project/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

Verify:

```bash
ls -lh ./smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

---

## 4. Environment Setup

We provide a minimal environment for reproducing the SMPLify optimization, evaluation metrics, ablation tables, and Plotly visualizations from the released OBJ/NPZ/keypoint files.

This minimal environment is not intended for rerunning the full EgoHumans data extraction, TokenHMR or 4D-Humans inference pipelines from raw images. 

Create the environment:

```bash
conda env create -f environment.yml
conda activate digitalhumans_team6
```

If `chumpy` fails during installation, install it manually after activating the environment:

```bash
python -m pip install --no-build-isolation chumpy==0.70
```
---

## 5. Make Scripts Executable

Run this once:

```bash
chmod +x *.sh
```

---

## 6. Generate Head Targets and 2D Keypoint JSON Files

The released data may already include the precomputed head_targets_unscaled/ and keypoints2d/ folders. Therefore, you can skip this session. 

However, if these folders are missing or you would like to try other frames, they can be generated from the EgoHumans ground-truth meshes and 2D pose files using the provided shell scripts.

### Generate Aria Head Targets

The head-target files are used by SMPLify-v1 and SMPLify-v2 as the ego-camera / head-position alignment targets. They are generated from the unscaled EgoHumans ground-truth meshes in data/mesh_cam_unscaled/<exo_cam>/rgb/<frame>/.

For one frame, run:

```bash
./extract_all_aria_targets.sh 00001 cam01
```

This creates one target file for each Aria identity:

```text
head_targets_unscaled/00001_aria01_cam01_egohumans_style.txt
head_targets_unscaled/00001_aria02_cam01_egohumans_style.txt
head_targets_unscaled/00001_aria03_cam01_egohumans_style.txt
head_targets_unscaled/00001_aria04_cam01_egohumans_style.txt
```

To generate targets for frames 00001 to 00006, run:

```bash
for FRAME in 00001 00002 00003 00004 00005 00006; do
    ./extract_all_aria_targets.sh "$FRAME" cam01
done
```

The script uses the unscaled GT meshes and saves the targets under head_targets_unscaled/, so the target coordinates remain consistent with mesh_cam_unscaled.

### Generate 2D Keypoint JSON Files

The 2D keypoint JSON files are used by SMPLify-v2 and the ablation experiments for 2D reprojection constraints. They are generated from EgoHumans 2D pose files stored under data/poses2d/<exo_cam>/rgb/.

For one frame, run:

```bash
./make_all_keypoints2d_json_frame.sh 00001 cam01
```

This creates one JSON file for each Aria identity:

```text
keypoints2d/00001_aria01_cam01.json
keypoints2d/00001_aria02_cam01.json
keypoints2d/00001_aria03_cam01.json
keypoints2d/00001_aria04_cam01.json
```

To generate keypoint JSON files for frames 00001 to 00006, run:

```bash
for FRAME in 00001 00002 00003 00004 00005 00006; do
    ./make_all_keypoints2d_json_frame.sh "$FRAME" cam01
done
```

The script reads the corresponding EgoHumans pose file, for example:

```text
data/poses2d/cam01/rgb/00001.npy
```

and writes the extracted per-person keypoints to keypoints2d/.


## 7. Reproduce Baseline Evaluation

The baseline evaluation compares the original HMR meshes against EgoHumans GT. The script evaluates frames `00001` to `00006`.

For 4D-Humans:

```bash
./run_baseline_all.sh 4dhumans
```

For TokenHMR:

```bash
./run_baseline_all.sh tokenhmr
```

Expected outputs:

```text
joint_baseline_4DHumans_00001_00006_auto_matched.txt
joint_baseline_TokenHMR_00001_00006_auto_matched.txt
```

This baseline script uses per-frame automatic identity matching based on root-aligned SMPL-joint MPJPE.

---

## 8. Reproduce SMPLify-v1 Translation Optimization

SMPLify-v1 optimizes only the global translation while keeping the initial SMPL pose and shape fixed.

Run:

```bash
./run_smplify_v1_frame.sh
```

This processes both TokenHMR and 4D-Humans for frames `00001` to `00006`.

Outputs:

```text
TokenHMR/demo_out/my_image_smplify_v1/
4D-Humans/demo_out/my_image_smplify_v1/
```

The script uses frame-specific detection-to-Aria mappings and SMPL ego-camera proxy vertices `2800` and `6260`.

---

## 9. Reproduce SMPLify-v2 Pose/Shape Refinement

SMPLify-v2 starts from the SMPLify-v1 results. It fixes the optimized translation and refines global orientation, body pose, and shape.

Run:

```bash
./run_smplify_v2_all_frames.sh
```

This processes both TokenHMR and 4D-Humans for frames `00001` to `00006`.

Outputs:

```text
TokenHMR/demo_out/my_image_smplify_v2/
4D-Humans/demo_out/my_image_smplify_v2/
```

---

## 10. Evaluate SMPLify-v1 and SMPLify-v2

The evaluator supports fixed detection-to-Aria mapping through the `--mapping` argument. Fixed mapping is preferred for reproducible comparison.

Run:

```bash
./evaluate_frames_00001_00006.sh
```

Expected merged outputs:

```text
metrics_00001_00006/tokenhmr_v1_00001_00006_fixed_mapping.csv
metrics_00001_00006/tokenhmr_v2_00001_00006_fixed_mapping.csv
metrics_00001_00006/4dhumans_v1_00001_00006_fixed_mapping.csv
metrics_00001_00006/4dhumans_v2_00001_00006_fixed_mapping.csv
```



---

## 11. Reproduce the Ablation Study

The ablation compares three optimization schedules:

1. `translation_only`: optimize global translation only.
2. `two_stage`: translation first, then fixed-translation pose/shape refinement.
3. `one_stage_all`: optimize translation, global orientation, pose, and shape jointly.

Run the ablation for 4D-Humans:

```bash
./run_v1_v2_and_ablation.sh 4dhumans
```

Run the ablation for TokenHMR:

```bash
./run_v1_v2_and_ablation.sh tokenhmr
```

By default, this script runs frames:

```text
00019, 00020, 00021, 00022, 00023, 00024, 00025
```

and uses frame-specific fixed detection-to-Aria mappings.

Expected outputs:

```text
metrics_ablation/4dhumans_00019_00025_three_pipeline_ablation_fixed.csv
metrics_ablation/4dhumans_00019_00025_three_pipeline_summary_all.csv
metrics_ablation/tokenhmr_00019_00025_three_pipeline_ablation_fixed.csv
metrics_ablation/tokenhmr_00019_00025_three_pipeline_summary_all.csv
```

---

## 12. Generate Visualizations

The visualization scripts generate interactive Plotly HTML files under:

```text
visualizations/
```

### Baseline vs. EgoHumans GT

```bash
./visualize_baseline_vs_egohumans.sh
```

Example outputs:

```text
visualizations/00001_TokenHMR_baseline_vs_GT_clean.html
visualizations/00001_4DHumans_baseline_vs_GT_clean.html
```

### Baseline vs. SMPLify-v1 vs. GT

```bash
./visualize_smplify_frame.sh
```

Example outputs:

```text
visualizations/00001_TokenHMR_baseline_vs_v1_vs_GT.html
visualizations/00001_4DHumans_baseline_vs_v1_vs_GT.html
```

### Baseline vs. SMPLify-v1 vs. SMPLify-v2 vs. GT

```bash
./visualize_smplify_v2_frame.sh
```

Example outputs:

```text
visualizations/00001_TokenHMR_baseline_vs_v1_vs_v2_vs_GT.html
visualizations/00001_4DHumans_baseline_vs_v1_vs_v2_vs_GT.html
```

Open any generated `.html` file in a browser.

---

## 13. Fixed Mapping Notes

For the main SMPLify-v1/v2 experiments on frames `00001` to `00006`, we use:

```text
00001: 0:aria03,1:aria02,2:aria01,3:aria04
00002: 0:aria02,1:aria03,2:aria04,3:aria01
00003: 0:aria03,1:aria02,2:aria04,3:aria01
00004: 0:aria03,1:aria04,2:aria02,3:aria01
00005: 0:aria03,1:aria04,2:aria02,3:aria01
00006: 0:aria04,1:aria03,2:aria02,3:aria01
```
For the ablation study on frames `00019` to `00025`, the mappings are defined inside `run_v1_v2_and_ablation.sh`.

---

