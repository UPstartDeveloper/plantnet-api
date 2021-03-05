# PlantNet API
A RESTful API that diagnoses plants using images of their leaves. Powered by Deep Learning.

## API Reference 
TBD

## How To Run Locally 
TBD


## Background

*PlantNet*: the convolutional neural network for this API was developed using transfer learning. The full implementation is available [in this Jupyter notebook](https://github.com/UPstartDeveloper/DS-2.4-Advanced-Topics/blob/main/Notebooks/Computer_Vision/Plant_Vision.ipynb).

If you wish to access the weights and biases of the CNN used in production, you can download them [here](https://github.com/UPstartDeveloper/DS-2.4-Advanced-Topics/tree/main/Notebooks/Computer_Vision/checkpoint) (same repo as above). Look for the files that start with `inception`.


## Deployment Guide: Heroku

This repository contains large files, so you might be inclined to use Git LFS. That's okay, however we need to keep in mind that this causes errors on Heroku (which won't be linked to the files tracked by Git LFS on default). To remedy this situation, please be sure to follow the instructions found [on this blog post](https://ankane.org/git-lfs-on-heroku) by Andrew Kane, which is a great guide. I would only make one small change to the directions Kane's blog, under the "Authentication" header. Instead of using the provided command to generate your SSH keys, use  this to give you more control over where the keys end up:

```
 ssh-keygen -C "<USERNAME>@<REPO_NAME>" -t rsa
 ```