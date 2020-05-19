# Topographic Convolutional Sparse Coding

![](./trained_models/RF.png)

This is an implementation of a convolutional sparse coding model with an extra linear layer on the
end, which results in a topographic ordering of filters.

## Run
To run the program:
```python
pip install -r requirements.txt
cd src/scripts
python train.py
```
To see a list of available hyperparameters to change:
```python
python train.py -h
```
A checkpoint of the model is saved every 10 epochs to `trained_models`. To see the tensorboard logs:
```python
tensorboard --logdir=runs
```

## References
* Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607–609. https://doi.org/10.1038/381607a0
* IMAGES.mat is downloaded from Olshausen's original Matlab implementation website: http://www.rctn.org/bruno/sparsenet/
