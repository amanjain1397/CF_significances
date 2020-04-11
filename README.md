# CF_significances
Collaborative Filtering on **MovieLens** dataset based on significances.

This is the pythonic implementation of Collaborative Filtering process which is based on significances of both users and items. The original paper referenced can be found [here](https://dl.acm.org/doi/10.1016/j.ins.2011.09.014).

### Data

For this project the **MovieLens 1M** dataset has been used, which can be found [here](https://grouplens.org/datasets/movielens/1m/). You can extract the zip file and put the *ratings.dat* file in the **./input/** folder.

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/amanjain1397/CF_significances.git
cd CF_significances
```
- Usage
```bash
python main.py --help
usage: main.py [-h] [--input [INPUT]] [--num_recomms NUM_RECOMMS]
               [--user USER] [--z Z] [--s_measures S_MEASURES] [--k K]

Colloborative Filtering based on significances

optional arguments:
  -h, --help            show this help message and exit
  --input [INPUT]       Input ratings.dat path (default: ./input/ratings.dat)
  --num_recomms NUM_RECOMMS
                        Number of recommendations to be made (default: 10)
  --user USER           User id for whom the recommendation has to be done
                        (default: 10)
  --z Z                 Number of items neighbors to be taken (default: 20)
  --s_measures S_MEASURES
                        similarity measure between users, one of "pearson",
                        "cosine" (default: pearson)
  --k K                 Number of user neighbors to be taken (default: 40)
```

### Working Example
First we must prepare the text file for the graph edgelist.
```bash
python main.py --input ./input/ratings.dat --num_recomms 10 --user 5 --z 20 --s_measures cosine --k 40 
Top 10 items recommended for user 5 are  [1264, 3365, 893, 900, 904, 931, 3392, 1020, 1031, 889]
```
