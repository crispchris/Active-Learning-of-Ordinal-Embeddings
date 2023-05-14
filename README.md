# Active Learning of Ordinal Embeddings: A User Study on Football Data

Code supplement for "Active Learning of Ordinal Embeddings: A User Study on Football Data", published in [TMLR](https://openreview.net/forum?id=oq3tx5kinu)

This code supports Active Learning user studies for relative comparisons. Data not included.

**Features**:
- Runs in the participant's browser
- Flexible Flask-based architecture
- Stores any information per annotator as a simple CSV
- Supports GPU training (comes with PyTorch)
- Uses gunicorn for parallel experiments (requires enough GPU VRAM)
- More features: randomized queries, query phases, warmup and tutorial, time keeping, stores annotations and skips, ..

**Limitations**:

- No explicit cloud support, e.g., limited scalability, usage of local storage.
- Bring-your-own-data-and-algorithms.

Related software: [NextML](http://nextml.org/) NEXT framework.

## Abstract


>Humans innately measure distance between instances in an unlabeled dataset using an unknown similarity function. Distance metrics can only serve as proxy for similarity in information retrieval of similar instances. Learning a good similarity function from human annotations improves the quality of retrievals. This work uses deep metric learning to learn these user-defined similarity functions from few annotations for a large football trajectory dataset.
We adapt an entropy-based active learning method with recent work from triplet mining to collect easy-to-answer but still informative annotations from human participants and use them to train a deep convolutional network that generalizes to unseen samples. 
Our user study shows that our approach improves the quality of the information retrieval compared to a previous deep metric learning approach that relies on a Siamese network. Specifically, we shed light on the strengths and weaknesses of passive sampling heuristics and active learners alike by analyzing the participants' response efficacy. To this end, we collect accuracy, algorithmic time complexity, the participants' fatigue and time-to-response, qualitative self-assessment and statements, as well as the effects of mixed-expertise annotators and their consistency on model performance and transfer-learning.


## Requirements

- [InfoTuple](https://github.com/Sensory-Information-Processing-Lab/infotuple) library published by the SIPLab at Georgia Tech in [AAAI 2020](https://arxiv.org/abs/1910.04115)
- Python 3.9 or higher, see environment.yml

## Running the Code

```
# create python environment
conda env create -f environment.yml

# run flask app
bash start_flask_locally.sh

# run http server that serves sample png
python -m http.server
```

## Citation

``` bibtex
@article{
    l{\"o}ffler2023active,
    title={Active Learning of Ordinal Embeddings: A User Study on Football Data},
    author={Christoffer L{\"o}ffler and Kion Fallah and Stefano Fenu and Dario Zanca and Bjoern Eskofier and Christopher John Rozell and Christopher Mutschler},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=oq3tx5kinu},
    note={}
}
```

## Additional Information 

- [Discussion on OpenReview](https://openreview.net/forum?id=oq3tx5kinu)
- [PDF on OpenReview](https://openreview.net/pdf?id=oq3tx5kinu)
- [Explanation on Youtube](https://youtu.be/xqOJAtjxjKE)

##  Contact

If you have any questions or comments about the code or data, please feel free to contact us.
