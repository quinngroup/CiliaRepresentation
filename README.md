# CiliaRepresentation

# Project Structure

## Directories
Top-level directories correspond to **modules** in the project. Specifically:

- **processing**
- **appearance**
- **dynamics**

Within a module directory, top-level files should include the main driver file. Subdirectories should include:

- **results**: a directory containing all experiment related information for the given module. Experiment data *must* include a `log.txt` file which documents the purpose of the experiment, the naming conventions employed by the experiment, and the corresponding conditions of each trial (e.g. learning rate of \[X] corresponds to lr\[X].py). Experiment data *may include* stdout capture (either via bash redirection or through the tee command), recorded weights, tf-board event files or anything else deemed relevant.
- **test_builds**: a directory containing all the various implementations and candidate models for the module. Each file in test_builds should be titled TB\[X] where \[X] is the next available index for test builds. See the test build section below for greater detail. 
- **experiments**: a directory containing bash scripts to run experiments and trials, as well as a manifesto detailing each experiment. Experiment scripts should be of form exp\[X].py where \[X] is the corresponding experiment index. See experiment section for greater detail.
- (optional) **scripts**: a directory containing any supplementary scripts. May be for testing, debugging or as a component of model operations.

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

1. An experiment should begin its life as an **issue**. The issue should be titled Exp\[X]: \[explanatory title] where \[X] is the next available experiment index. That is to say if the most recent experiment was Exp17, then the new experiment should be titled Exp18: \[explanatory title]. Generally they will begin as a vague goal (e.g. explore different options for learning rate). Such experiments should be tagged `design`.

2. Design stage experiments should be refined by either the originator of the issue or any other members of the project (e.g. grid-search different learning rate values over \[1e-5 through 9e-5]). After being refined, the tag should be changed to `script-ready`

3. Then, a script must be made for the experiment. The script should be stored in the experiments directory following the conventions mentioned above. After a script is made, the manifesto file in the experiments directory must be updated to include the details of the newly added edxperiment. Finally the issue tag must be updated and set to `trial-ready`

4. Trial-ready scripts are to be executed on any suitable hardware choice. The primary choice is GCP cloud servers; however, alternative hardware choices are acceptable as well (e.g. a particularly well equipped desktop).

5. Once the trial has been completed as specified by the experiment, the issue must be closed (unless further discussion is required).


## Test Builds 
TBD
