# BALSAM
## A django based platform for supervised feature extraction and visualization of Multi Capillary Column - Ion Mobility Spectrometry (MCC-IMS) data.

BALSAM is available at https://exbio.wzw.tum.de/balsam/ .

## Run your own instance
We host a public docker image for **local** deployment on dockerhub. It includes nginx and runs django using uwsgi. Use 

```
sudo docker pull philmaweb/balsam_docker:local
new_id=$(sudo docker run -d -p 80:80 philmaweb/balsam_docker:local)
```
to deploy balsam on localhost `port 80`. Visit `127.0.0.1` in your favorite webbrowser.
Log into the container using `sudo docker exec -it --env COLUMNS=`tput cols` --env LINES=`tput lines` "$new_id" bash`. 

## Setup instructions
Not recommended - use the docker container instead. See docker container `/home/django/start_local.sh`

Based on the guide on https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04 : (ubuntu 18.04 also works). Also needs rabbitmq, supervisor, nginx and celery.