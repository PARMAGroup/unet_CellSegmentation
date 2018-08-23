# unet_CellSegmentation
Pre trained Unet  for cell segmentation

The directory tree is organized as follows:

    - ./raw
    Raw images  used for prediction have to be stored here. 
    
    - ./weights
    Trained weights for Unet are stored here.

    - ./preds
    The prediction results are stored here as output images.

    - ./unet_CellSegmentation.py
    This is the main program which loads the image dataset, preprocess it and then performs the predictions


Instructions:

- To run use this command:
     python unet_CellSegmentation.py

- Requirments:
  - Python >=3.5 with standard modules (numpy, etc)
  - tensorflow
  - keras
  - pillow

