
### Purpose
ReAgent has a layered architecture. If you want to change any module, you need to find and make changes to existing code. Therefore I have created my own pipeline so that its becomes a breeze to customize data pipeline, model manager, net builder and config file.

### Objectives
The objectives of the modifications were to make it easier to:

1. Use custom config files and data pipelines
2. Experiment with custom models
3. Minimize changes in existing files
4. and original code can work as is

### Modifications
#### New Files
1. reagent/workflow/train.py: This file is contains the pipeline for building model manager, net builder, data loader, config file reader, trainer and evaluator.
2. reagent/data/my_data_module.py : This file implements pl.LightningDataclass derived from ReAgentDataModule.
3. reagent/data/myloader.py : This is implements torch Dataset class. I have provided dummy functions to explain dataframe format required by ReAgent. You need to create your own ETL here.
4. reagent/model_managers/discrete/mobile_dqn.py : This file contains thecustom model manager.
5. reagent/net_builder/discrete_dqn/mob_net.py: This file implements custom net builder.
6. reagent/models/mobnet.py : This file contails custom model.
7. reagent/workflow/sample_configs/my_config.json: This is custom config file written in JSON format. you can use any config file and change the reader in train.py file. 

#### Modifications to existing files
1. reagent/net_builder/unions.py: Added entry for my custom netbuilder.
2. reagent/workflow/utils.py: If you want to restart from previous checkpoint. Uncomment "resume_from_checkpoint". If there is no existing checkpoint file, it will through error.

### Usage
>cd ReAgent
>python reagent/workflow/train.py --cfg reagent/workflow/sample_configs/my_config.json 

