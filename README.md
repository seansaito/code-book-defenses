# code-book-defenses
Evaluation of adversarial robustness of neural networks that use MSE and code-book target representations

# Setting Up

We use Python 3.x. Install the dependencies via:

```bash
$ pip3 install -r requirements.txt
``` 

Note that we depend on `tensorflow-gpu==1.19.0`. This may not work depending on your system's configurations and library/driver versions (e.g. CUDA, CuDNN).
Please install the appropriate TensorFlow version.

# Running training & attacking the model

* The `main.py` module trains a DenseNet model and runs a number of untargeted and targeted attacks on it during test time.
* The `JSON` files in `configs/` are the configurations of the DenseNet model. They specify parameters such as architecture of the model,
dataset to train on, etc.
* `code_book_defenses/config.py` includes several variables and dictionaries which configure the attack experiments.
* For example, the lists `untargeted_attacks` and `targeted_attacks` indicate which attacks to run on the trained model.
* The `attack_name_to_params` dictionary enumerates the parameters for each attack. You will notice that at most one field from every attack dictionary
is a list. For each attack during test time, we iterate over each value of the list to construct adversarial examples.

After you have installed the dependencies and identified the configurations you want to run, you can trigger training via the following:

```bash
$ python3 main.py --config=run_configs/name_of_config.json --gpus=0 --experiment=experiment_1
```

The arguments for `main.py` are as follows:

* `config` (required): The path to the configuration `JSON` file
* `gpus`: GPUs to use (NOTE: the current implementation is not optimized for multi-gpu training)
* `experiment`: Name of the experiment (can be some unique identifier)

