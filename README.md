# SentimentMT

**Summary**

This is the repo associated with the paper [**Sentiment-based Candidate Selection for NMT**](https://arxiv.org/abs/2104.04840), co-written by me (Alex Jones) and my supervisor [Derry Wijaya](https://derrywijaya.github.io/web/). The paper describes a decoder-side approach for selecting the translation candidate that best preserves the automatically-socred sentiment of the source text. To this end, we train three distinct sentiment classifiers: an English BERT model, a Spanish XLM-RoBERTa model, and an XLM-RoBERTa model fine-tuned on English but used for sentiment classification in other languages, such as French, Finnish, and Indonesian. We compute a softmax over the logits returned by these classifiers to obtain the probability of a text <img src="https://latex.codecogs.com/svg.image?t" title="t" /> being in the positive class, and call this number the "sentiment score":

<img src="https://latex.codecogs.com/svg.image?S(t)&space;=&space;P(c=1&space;\mid&space;t)" title="S(t) = P(c=1 \mid t)" />

We then generate <img src="https://latex.codecogs.com/svg.image?n" title="n" /> translation candidates using beam search and select the candidate <img src="https://latex.codecogs.com/svg.image?y" title="y" /> whose sentiment score differs least from that of <img src="https://latex.codecogs.com/svg.image?t" title="t" />:

<img src="https://latex.codecogs.com/svg.image?y&space;=&space;argmin_{x&space;\in&space;X}|S(x)&space;-&space;S(t)|,&space;|X|=n" title="y = argmin_{x \in X}|S(x) - S(t)|, |X|=n" />

We conduct human evaluations on English-Spanish and English-Indonesian translations with proficient bilingual speakers and report the results in our paper. We also provide examples of tweets translated using this method in the Discussion and the Appendix. 

**Dependencies**

[PyTorch](https://pytorch.org) \
[Transformers](https://huggingface.co/transformers/installation.html)\
[Scikit-learn](https://scikit-learn.org/stable/install.html)\
[SciPy](https://www.scipy.org/install.html)\
[BeautifulSoup](https://pypi.org/project/bs4/) (for text preprocessing)\
[Numpy](https://numpy.org/install/)\
[Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)

**Sentiment Classification**

We construct sentiment classifiers by fine-tuning on labeled sentiment data in English and Spanish separately. The English-only sentiment classifier is constructed using BERT; the notebook for training is available [here](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Sentiment%20Classifier%20Notebooks/English_sentiment_notebook.py) and is based on the [BERT fine-tuning tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) by Chris McCormick and Nick Ryan (as are all the notebooks we used for training our sentiment classifiers???citations are provided in-notebook). We also [fine-tune XLM-RoBERTa](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Sentiment%20Classifier%20Notebooks/Spanish_sentiment_notebook.py) using annotated Spanish data, and then [again](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Sentiment%20Classifier%20Notebooks/Multilingual_sentiment_notebook.py) using the English sentiment data. The sentiment models themselves (the PyTorch files containing the parameters) are available [here](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Sentiment%20Models%20(Download%20Links)/Sentiment%20Models%20(Links%20to%20Downloadable%20PyTorch%20Files).rtf), and the annotated sentiment data is available at the following links:

* [English train](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Data%20and%20Reference%20Materials/Sentiment%20Train%20Data/English_train.rtf), [English test](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Data%20and%20Reference%20Materials/Sentiment%20Test%20Data/English_test.csv)
* [Spanish train tweets](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Data%20and%20Reference%20Materials/Sentiment%20Train%20Data/Spanish_train_tweets.rtf), [Spanish train labels](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Data%20and%20Reference%20Materials/Sentiment%20Train%20Data/Spanish_train_sentiments.csv), [Spanish test](https://github.com/AlexJonesNLP/SentimentMT/blob/main/Data%20and%20Reference%20Materials/Sentiment%20Test%20Data/Spanish_test.csv)

**MT**

We perform machine translation using the open-source [Helsinki-NLP/OPUS-MT models](https://github.com/Helsinki-NLP/Opus-MT), which offers pretrained models for easy usage [here](https://huggingface.co/Helsinki-NLP). We opted for this system because we were easily able to generate n-best lists and incorporate sentiment-based selection into the decoding step. Because we used pretrained models, we don't perform any of our own training, but [these notebooks](https://github.com/AlexJonesNLP/SentimentMT/tree/main/MT%20Notebooks) show how we integrate sentiment scoring into the translation selection process.
Another advantage of the Helsinki-NLP models was the wide variety of supported languages, which we wielded to our advantage in trying our approach on many different languages (see the Appendix of our paper for concrete examples).

**Experimental Materials**

In human evaluations of the translations, we asked participants to grade translations based on both their accuracy (broadly speaking) and their level of sentiment divergence, and also asked them to provide reasons *why* they thought the sentiment of the source text differed from that of the translation, if applicable. We performed both an English-Spanish and English-Indonesian evaluation. See the following files for reference:
* The [translations that were evaluated](https://github.com/AlexJonesNLP/SentimentMT/tree/main/Data%20and%20Reference%20Materials/Translations%20for%20Evaluation)
* [Source texts](https://github.com/AlexJonesNLP/SentimentMT/tree/main/Data%20and%20Reference%20Materials/Idiomatic%20Source%20Texts) (English tweets) deemed to be particularly "idiomatic"
* The [evaluation templates](https://github.com/AlexJonesNLP/SentimentMT/tree/main/Data%20and%20Reference%20Materials/Evaluation%20Templates) themselves
* The [notebooks](https://github.com/AlexJonesNLP/SentimentMT/tree/main/Evaluation%20Analysis%20Notebooks) we used in analyzing the results of the human evaluations

**License**

[BSD 3-clause](https://github.com/AlexJonesNLP/SentimentMT/blob/main/LICENSE).

**Citation**

Please cite our paper if you use any of the resources in this repo for your research:

```
@inproceedings{jones-wijaya-2021-sentiment,
    title = "Sentiment-based Candidate Selection for {NMT}",
    author = "Jones, Alexander G  and
      Wijaya, Derry",
    booktitle = "Proceedings of the 18th Biennial Machine Translation Summit (Volume 1: Research Track)",
    month = aug,
    year = "2021",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2021.mtsummit-research.16",
    pages = "188--201"}
```
