# cifar_prediction_app

Cifar10 / Cifar100 estimation with web API to input own images for prediction

## Summary

This is an extension of the Tensorflow's deep CNN tutorial. There several contributions in this work:

 - dropout layer is added to the CNN
 - the scripts are adapted for estimation using CIFAR-100 as well as CIFAR-10 dataset
 - web API (uses `flask` can be used to insert own pictures for prediction)
 
## How to launch the application
 
**Training**

To train the CNN network run
 
```
python codes/cifar_prediction_app.py
```
 
Script will automatically download the data into the `data/` folder and start the estimation.
Edit the

```
tf.app.flags.DEFINE_integer('max_steps', 150000, ...
```

to change maximum number of iterations.

**TensorBoard**

Log files neccessary for running TensorBoard are saved in the `train/` folder.
One can therefore run the TensorBoard using e.g.

```
cd C:/codes/cifar_prediction_app
tensorboard --logdir=train
```
 
**Prediction**
 
To run the prediction app run

```
python codes/cifar_prediction_app.py
```

You should immediatelly see a web form prompting you to upload an image for classification. `.jgp` or `.png` can be used. After successfully uploading the image you should see a table with estimated probabilities that an object on the image belongs to a class.

Note certain amount of memory has to be available for both estimation and prediction of the CNN. 
