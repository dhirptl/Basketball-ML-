1. Copy this folder to e.g. `court_pose_dataset/`.
2. Create layout:
   - `court_pose_dataset/train/images/` and `court_pose_dataset/train/labels/`
   - `court_pose_dataset/valid/images/` and `court_pose_dataset/valid/labels/`
   (same image basename in images/ and labels/, e.g. `frame001.jpg` + `frame001.txt`).
3. Class 0 = Court. Each label line: class id, then cx cy w h (bbox), then 8*(x y v) in normalized 0..1.
4. Keypoint index order (see project README.md).
5. Run: `python -m src.validate_pose_dataset --labels-dir court_pose_dataset/train/labels`
6. Run: `python -m src.train_court_pose --data court_pose_dataset/data.yaml`
