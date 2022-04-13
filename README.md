# savae

To prepare the settings for each experiment, a config file with name format config_[config_id].py needs to be added to configs directory. As shown in the provided template, the the path to the dataset, and also all hyper-parameters can be set in this file. Later on, to run the main python scripts, the id of the config file will be passed to specifiy the experiment configuration.
-----
To train the model using subgroup-aware VAE, use the following command:

**python train --config_id [config_id]**

----

To generate latent representation of the input data:

**python generate-latent-reps.py --config_id [config_id]**

-----

To mine subgroups based on the latent representation:

**python mine_subgroups.py  --config_id [config_id]**

-----
To generate latent traversal images:

**python traverse_latents.py --config_id [config_id]**

------

To generate average image of whole population, and each subgroup:

**python visualize_sgds.py --config_id [config_id]**
