# Cuneiform Classification Project

## Configuration
1. Go to configs/default.yaml
2. Adjust paths if necessary 
3. Can adjust settings for dataset, model, train and eval if wanted

## Main Run
1. Main file is go_time.ipynb 
2. Run all will prepare dataset, train, validate, test, and produce visuals
3. For a per-image inference, go to inference.ipynb

## Inference Run
1. From test_image_names.csv in main project folder, choose one of the image names
2. Place best_map50.pth in main project folder
3. Go to inference.ipynb
4. In paths, replace the image_name with the one you have chosen. Make sure to include the file extension
5. Run all 
6. Visuals will produce a copy of the chosen image with bounding boxes marked on it. Yellow is the ground truth, Green is the True Positive predictions, and Blue is the False Positive predictions 
7. This image will also be saved as a png file
