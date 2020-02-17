# CiliaRepresentation

# Project Structure

## Directories
Top level directories include:

- **VTP**: the main directory under which the project code is stored
- **results**: a directory containing all experiment related information for the given module. Experiment data *must* include a `log.txt` file which documents the purpose of the experiment, the naming conventions employed by the experiment, and the corresponding conditions of each trial (e.g. learning rate of \[X] corresponds to lr\[X].py). Experiment data *may include* stdout capture (either via bash redirection or through the tee command), recorded weights, tf-board event files or anything else deemed relevant.
- **experiments**: a directory containing bash scripts to run experiments and trials, as well as a manifest detailing each experiment. Experiment scripts should be of form exp\[X].py where \[X] is the corresponding experiment index. See experiment section for greater detail.

VTP subdirectories correspond to **modules** in the project. Specifically:

- **processing**
- **appearance**
- **dynamics**
- **utils**: not technically a module, but stores scripts and code to be utilized in other modules

Within a module directory, top-level files should include the main driver file. Subdirectories should include:

- **test_builds**: a directory containing all the various implementations and candidate models for the module. Each file in test_builds should be titled [3 letter initialism]\[X] where \[X] is the next available index for test builds. See the test build section below for greater detail. 

## Branches
Certain dependencies exist between modules which can make parallel development challenging, such as the fact that the Appearance module may depend on results from the Processing module; however, these do not completely restrict the development of modules in parallel. Parallel development is facilitated through branching.
We use the following branches:

- **master**: contains the most recent *stable* build of the entire pipeline. 
- **candidate**: contains prototype pipeline composed of (hopefully) stable modules. Used to test module integration strategies and pipeline stability as a whole
- **processing**: contains prototype of processing module.
- **appearance**: contains the most recent *stable* build of the processing module and a prototype of the appearance module.
- **dynamics**: contains the most recent path dataset generated by the most recent *stable* appearance module (may optionally contain that module)

Child/sibling branches may spawn to sub-prototype any of the above branches (e.g. if there are two separate candidates that need to be tested)

## Experiments
Experiments can be over basically anything, ranging from stability (not crashing and burning after a single epoch) to hyperparameter tuning through loss optimization. Here is the general experiment lifestyle:

1. An experiment should begin its life as an **issue**. The issue should be titled Exp\[X]: \[explanatory title] where \[X] is the next available experiment index. That is to say if the most recent experiment was Exp17, then the new experiment should be titled Exp18: \[explanatory title]. Generally they will begin as a vague goal (e.g. explore different options for learning rate). Such experiments should be tagged `experiment design`.

2. Design stage experiments should be refined by either the originator of the issue or any other members of the project (e.g. grid-search different learning rate values over \[1e-5 through 9e-5]). After being refined, the tag should be changed to `script ready`

3. Then, a script must be made for the experiment. The script should be stored in the experiments directory following the conventions mentioned above. After a script is made, the manifest file in the experiments directory must be updated to include the details of the newly added experiment. Finally the issue tag must be updated and set to `trial ready`

4. Trial-ready scripts are to be executed on any suitable hardware choice. The primary choice is GCP cloud servers; however, alternative hardware choices are acceptable as well (e.g. a particularly well equipped desktop). When running a script, say `exp1.sh`, go to the experiments directory and type `nohup bash exp1.sh & > ../results/exp1/out.txt`

5. Once the trial has been completed as specified by the experiment, the issue is then set to the `analysis` tag. At this point, the results are observed and analyzed, with the discussion and/or conclusion being recorded on the ticket.

6. Finally, the issue is closed.


## Test Builds 
Although each module can potentially have test builds, the processing module will most likely not require a rigorous test build process, and hence may be exempt from following this structure. However, the other two modules should implement test builds in the very appropriately named **test_builds** subdirectory of the respective module. Test builds will follow a similar lifecycle to experiments, and may well be considered a special type of experiment. In particular, here is the lifecycle:

1. A test model should begin its life as an **issue**. The issue should be title \[3 Letter Initialism\]\_\[index number\]:\[Full Title\]. Take, for example, the Naturalized VampPrior model. Suppose that we wanted to propose an initial minimal implementation of the model. We'll take the 3 letter initalism to be NVP. Then, the issue title should be `NVP_0:Naturalized VampPrior`. Note the index is 0 because it is a proposal for an *initial* implementation. From here on out, the next proposed variation to NVP would be recorded as `NVP_1:...`. For example, suppose we wanted to propose a modification to the shape of the posterior distributions. Then a viable name could be `NVP_1: Altered Posterior Distributions`. The issue's tag should be set to `model design`

2. Design progress should be recorded on the original issue and continued until a viable formulation has been reached. Then the issue tag should be set to `implementation` for model implementation.

3. Next, a python file should be created implementing the model. After this is done, the original issue should be closed. Any further work on the model will either be architectural changes which will be reflected in other test build issues, or as experiments as discussed above. The test build issue may be referenced in other issues, either test build or experiments, to provide a singular history for the model.

## Distributed Mode

Run using `python -m torch.distributed.launch --nproc_per_node=[#GPUs, default 2] Driver.py [args]`. Note that when running in distributed mode, the `--batch_size` argument refers to effective batch-size, meaning that each GPU will receive `args.batch_size/#GPUs` per batch. As an example, running on 2 GPUs with batch-size=80 will run each GPU with batch-size=40
