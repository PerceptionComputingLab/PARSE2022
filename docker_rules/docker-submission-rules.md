





Please name your docker image as your teamname, and the tag is latest, then save your docker image as teamname.tar.gz, send us a link to download the tar file. Note that, the docker image name requires lowercase letters. You can use the following command to generate docker tar file.

```bash
docker save teamname:latest –o teamname.tar.gz
```

when we receive your docker tar file, we will run your program with commands as follows

```bash
docker load < teamname.tar.gz
```

```bash
docker run --gpus "device=0" –name teamname –v /home/amax/Desktop/input:/input –v /home/amax/Desktop/predict:/predict teamname:latest
```

we will mount  /home/amax/Desktop/input(a folder contains all CT files for testing) to /input in your docker container, and mount /home/amax/Desktop/predict(an empty folder used to save segmentation file) to /predict in your docker container.

Your program in your docker container should do the following things:

- obtain each CT file(.nii.gz) in folder /input
- apply your segmentation algorithmn to segment the target
- save the segmentation mask to /preidct. Note that, the filename of segmentaion mask file should be the same as the CT file,  the segmentation mask is a 3D zero-one array(0 stands for background,1stands for target), the meta information of the segmentation mask file should be consistent with that of original CT file.

If you don't kown how to build a docker image for your program, you can refer to the video https://youtu.be/wkHUtOCEHro and the example https://github.com/heyingte/build-docker-image-example







