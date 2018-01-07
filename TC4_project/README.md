# TC4 project
The goal is to design a model to correct typos in texts without a dictionaries.

In this problem, a state refers to the correct letter that should have been typed, and an observation refers to the actual letter that is typed. Given a sequence of outputs/observations (i.e., actually typed letters), the problem is to reconstruct the hidden state sequence (i.e., the intended sequence of letters). Thus, data for this problem looks like:

[('t', 't'), ('h', 'h'), ('w', 'e'), ('k', 'm')]
 [('f', 'f'), ('o', 'o'), ('r', 'r'), ('m', 'm')] 
The first example is misspelled: the observation is thwk while the correct word is them. The second example is correctly typed.

Data for this problem was generated as follows: starting with a text document, in this case, the Unabomber's Manifesto, which was chosen not for political reasons, but for its convenience being available on-line and of about the right length, all numbers and punctuation were converted to white space and all letters converted to lower case. The remaining text is a sequence only over the lower case letters and the space character, represented in the data files by an underscore character. Next, typos were artificially added to the data as follows: with 90% probability, the correct letter is transcribed, but with 10% probability, a randomly chosen neighbor (on an ordinary physical keyboard) of the letter is transcribed instead. Space characters are always transcribed correctly. In a harder variant of the problem, the rate of errors is increased to 20%.

Data folders contains 4 pickles: train10 and test10 constitute the dataset with 10% or spelling errors, while train20 and test20 the one with 20% or errors.

## How to use

We have realized HMM one order and two order(HMM1 and HMM2). You can use code by notebook and python file(The code in notebook and python file are same).

### All codes are writen by python3.6

### To install necessay libraries

```shell
pip install -r requirements.txt
```

### If you want to read code, it's better to use notebook.
```shell
jupyter notebook
```

### If you just to see the results, just run python file,
```shell
python HMM1.py
python HMM2.py
```
