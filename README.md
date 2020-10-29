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
Not recommended - use the docker container instead. See docker container `/home/django/start_local.sh`.

Based on the guide from https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04 : (We migrated to Ubuntu 18.04). Also relies on rabbitmq, supervisor, nginx and celery.

### License
`BALSAM` is licensed under GPLv3. It makes use of [BreathPy](https://github.com/philmaweb/BreathPy), which is itself licensed under GPLv3, but contains binaries for PEAX, which is a free software for academic use only.
See
> [A modular computational framework for automated peak extraction from ion mobility spectra, 2014, D’Addario *et. al*](https://doi.org/10.1186/1471-2105-15-25)

## Contact
If you run into difficulties using BALSAM, please open an issue at our [GitHub](https://github.com/philmaweb/balsam_django) repository. Alternatively you can write an email to [Philipp Weber](mailto:pweber@imada.sdu.dk?subject=[BALSAM]%20BALSAM).

## Citation

BALSAM has been published. Please cite:
> [Philipp Weber, Josch Konstantin Pauling, Markus List, and Jan Baumbach. "Balsam—an interactive online platform for breath analysis, visualization and classification." Metabolites 10, no. 10 (2020): 393.]