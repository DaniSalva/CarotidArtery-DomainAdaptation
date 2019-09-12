Instructions to build docker image for image segmentation code
==============================================================

Requeriments
	nvidia-docker installed

Files-folders needed
	/docker_templates
	/SemSeg

STEPS

		1- Build docker images.

			docker build -f docker_templates/Dockerfile -t image_base .

			docker build -f docker_templates/Dockerfile -t image_sem_seg .

		2- Create docker virtual volume (container persistence)

			docker volume create --name semsegvolume

		3- Run docker container mapping nvidia drivers and docker volume with "keras_semantic_segmentation" folder.

			 nvidia-docker run --rm -it --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 -v nvidia_driver_367.57:/usr/local/nvidia:ro  -v semsegvolume:/SemSeg/keras_semantic_segmentation -it image_sem_seg /bin/bash

		4- For entering the container afterwards.

			docker exec -it "container_ID" /bin/bash


		5- Modify if necessary train.sh --> Vim editor available !!

			PYTHONPATH=/SemSeg/keras:/SemSeg/multimodal_keras_wrapper:/SemSeg/keras_semantic_segmentation:$PYTHONPATH THEANO_FLAGS='device=cuda1,optimizer=fast_compile,optimizer_including=fusion' python -u main.py

			Specially the argument device=cuda1 (you must select the device id of your GPU).


		6- Configure config.py for training --> Vim editor available !!

		7- Run the image segmentation training

			nohup ./train.sh &> log_training.out&