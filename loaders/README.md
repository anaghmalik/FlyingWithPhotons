1. Under `<scene_name>/training_files`, you will find the files required for both training and evaluation.

- `samples.h5` are the precomputed batches shaped (n_views*$512^2$, n_bins, n_channels), where n_views*$512^2$ is the total number of pixels in the dataset. 
- `x.h5`, `y.h5` and `file_indices.h5` contain the x coordinate of the pixel, y coordinate of the pixel and the index of the view the pixel belongs to, the index is the same as in `transforms_train.json` 


2. Transforms files.

- `transforms_train.json` contains the transforms used in training (the order matters)
- `transforms_test.json` are the transforms used for visualization during training 
- `transforms_evaluation.json` contain the transforms used for evaluation, the appropriate measured/ground-truth transient h5 files can be found also in the `training_files` directory. 
- `transforms_bezier.json` and `transforms_spiral.json` contain cameras along trajectories to make videos with changing time and camera position, there is no reference transients for these. 

Each transforms file contains a list `"frames"`, the list captures the name and extrinsics for each camera. 
The parameter `"camera"` encodes the intrinsics, either in focal length of 


3. Names and notation.  

- The file names for the transients are in the format `{i}_{j}`, where `i` indicates the elevation level (there are 3 elevation levels, you can find the exact degrees for each scene in the paper supplement) and j indicates the rotation, for the captured dataset an increment of `j` is a 6 degree rotation. 


4. Downloading.

- The official code release has a download script in `misc/download_data.py`. However you can also download the data yourself either through the Dropbox download button, or by right-clicking a folder and selecting copy link address, then

```
wget "copied link" 
```

will start a download of the folder. 


5. Images. 

Under `<scene_name>/images` you will find the integrated transients (these are not the ones used in the evaluation script), check `eval.py` to see how we get those. 

For the Kennedy and Coke scene you will also find the `rgb_images` folder with rgb images for the captured viewpoints. 


6. Direct/Indirect. 

The root contains `pulse.py` which has the impulse response.
Additionally `statue/direct` and `statue/indirect` contain the files for the transients separated into direct and indirect components. 