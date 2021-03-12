# PlantNet API
A RESTful API that diagnoses plants using images of their leaves. Powered by Deep Learning.

## API Reference 
TBD

## How To Run Locally 
### Option 1: Using Python `virtualenv`
You can use the following commmand to create a new Python 3 virtual environment (any version of Python 3.5–3.8 should work, because we are using Tensorflow 2):
```
python3 -m venv env
``` 
Afterwards you can install the dependencies via `pip`:
```
(env) python3 -m pip install -r requirements.txt
```
Finally you can configure the environment variables and run the Flask module:
```
(env) export FLASK_ENV=development; export FLASK_APP=application; flask run
```
### Option 2: Using Docker
TBD


## Background

*PlantNet*: the convolutional neural network for this API was developed using transfer learning. The full implementation and training is available [in this Jupyter notebook](https://github.com/UPstartDeveloper/DS-2.4-Advanced-Topics/blob/main/Notebooks/Computer_Vision/Plant_Vision.ipynb).

If you wish to access the weights and biases of the CNN used in production, you can download them in the [`/plantnet` directory](https://github.com/UPstartDeveloper/plantnet-api/tree/main/plantnet) of this repository.


## Deployment Guide: Heroku

This repository contains large files, so you may choose to use Git LFS (don't need to however). If you do choose to do so, we need to keep in mind that this causes errors on Heroku (which doesn't have Git LFS installed by default). To remedy this situation, please be sure to follow the instructions found [on this blog post](https://ankane.org/git-lfs-on-heroku) by Andrew Kane, which is a great guide. I would only make one small change to the directions Kane's blog, under the "Authentication" header. Instead of using the provided command to generate your SSH keys, use  this to give you more control over where the keys end up:

```
 ssh-keygen -C "<USERNAME>@<REPO_NAME>" -t rsa
 ```