# BeeAmI
The beestly accurate algorithm for predicting what monkey do.<br>

Parts of the code were generated with the help of Github Copilot.<br>

To add a model that uses angle classification, add an `angle` output to the `positionEstimator` function.<br>

Running test script:<br>
`testScript(`<br>
`model: (string) folder name of the model under /BeeAmI`<br>
`visualize: (float) delay_for_plotting i.e. 0 for no plotting, 1000 for 1s`<br>
`granular: (boolean) to plot each trial or not to plot, that is the question`<br>
`seed: (int) chosen rng config`<br>
`validation: (boolean) whether used for validation`<br>
`folds: (int) number of cross-validation folds)`<br>

Running validation script:<br>
`validationScript(`<br>
`models: ({string}) folder name of the models under /BeeAmI`<br>
`runs: (int) number of times testScript is run`<br>
`basic_visualisation: (boolean) whether to show the main visualisation plot`<br>
`folds: (int) number of folds)`<br>

# Models
Each in their own folder under `/BeeAmI`.<br>

# Plots
In the root folder.<br>

# Results
In `/results`.<br>
