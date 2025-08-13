This project tests FFNN and RNN PyTorch programs for sentiment analysis tasks.
JSON data for training, validation and test sets is provided and the code utilizes the training
and validation data to train the model parameters and tests on the test data.

One example on running the code:

**FFNN**

```shell
$ python3 ffnn.py --hidden_dim 8 --epochs 10 --train_data ./training1.json --val_data ./validation.json --test_data ./test.json
```
**RNN**
```shell
$ python3 rnn.py --hidden_dim 8 --epochs 10 --train_data training2.json --val_data validation.json --test_data test.json
```
