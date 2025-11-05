RCLP Medical Scripts - README
=============================

Included files:
- chexpert.py      : Main RCLP-style training script for CheXpert-format datasets.
- NIH.py           : Wrapper that calls chexpert.train_rclp for NIH dataset inputs.
- example_tasks.json : Example tasks JSON showing how to define ordered tasks.
- RCLP_README.txt  : This file.

Quick start (example):
1) Prepare a metadata CSV for your dataset with columns:
   - image_path : relative path to image under --data-dir
   - labels     : semicolon-separated label names present in the image
   Example row: "images/patient1_0001.png","Atelectasis;Cardiomegaly"

2) Prepare a tasks JSON mapping task IDs to lists of class names (see example_tasks.json). Example format:
   {
     "task1": ["Atelectasis", "Cardiomegaly"],
     "task2": ["Effusion", "Infiltration", "Pneumonia"],
     ...
   }

3) Run training (CheXpert example):
   python chexpert.py --data-dir /path/to/chexpert/images --meta-csv /path/to/meta.csv --tasks-json /path/to/tasks.json --output-dir ./outputs --batch-size 16 --epochs-per-task 10

   Run NIH similarly using NIH.py:
   python NIH.py --data-dir /path/to/nih/images --meta-csv /path/to/nih_meta.csv --tasks-json /path/to/tasks.json --output-dir ./outputs

Notes & Tips:
- Both scripts assume you already have the image files locally (CheXpert and NIH typically require registration to download).
- If you run on Kaggle, set --output-dir to /kaggle/working to preserve outputs.
- Adjust --num-workers according to your environment (0-8 typical).
- The scripts compute per-class thresholds on validation sets for pseudo-labeling; ensure validation split has some positive examples for each class where possible.
- If you want a smaller model for faster experimentation, modify DenseNet121Feature to use a lightweight backbone (e.g., torchvision.models.mobilenet_v2).

If you want, I can add:
- A helper to auto-generate the meta CSV from original dataset label files (if you provide them).
- Template scripts to visualize per-task metrics (F1, AUC) and confusion matrices.
- A SLURM / GPU cluster-friendly runner script.
