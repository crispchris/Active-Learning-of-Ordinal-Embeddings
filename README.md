# Active Learning of Ordinal Embeddings: A User Study on Football Data

Code supplement for "Active Learning of Ordinal Embeddings: A User Study on Football Data", appearing in [TMLR](https://openreview.net/forum?id=oq3tx5kinu)

## Paper abstract


Humans innately measure distance between instances in an unlabeled dataset using an unknown similarity function. Distance metrics can only serve as proxy for similarity in information retrieval of similar instances. Learning a good similarity function from human annotations improves the quality of retrievals. This work uses deep metric learning to learn these user-defined similarity functions from few annotations for a large football trajectory dataset.
We adapt an entropy-based active learning method with recent work from triplet mining to collect easy-to-answer but still informative annotations from human participants and use them to train a deep convolutional network that generalizes to unseen samples. 
Our user study shows that our approach improves the quality of the information retrieval compared to a previous deep metric learning approach that relies on a Siamese network. Specifically, we shed light on the strengths and weaknesses of passive sampling heuristics and active learners alike by analyzing the participants' response efficacy. To this end, we collect accuracy, algorithmic time complexity, the participants' fatigue and time-to-response, qualitative self-assessment and statements, as well as the effects of mixed-expertise annotators and their consistency on model performance and transfer-learning.

## Requirements

- [InfoTuple](https://github.com/Sensory-Information-Processing-Lab/infotuple) published by the SIPLab at Georgia Tech in [AAAI 2020](https://arxiv.org/abs/1910.04115)
- Python 3.7 or higher
- see requirements.txt

## Running the Code

The code is to be published here.

```
pip install -r requirements.txt
```

## Citation

``` bibtex
citation TBD
```

## Contact

If you have any questions or comments about the code or data, please feel free to contact us.
