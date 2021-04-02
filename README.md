# SentimentMT

This is the repo associated with the paper **Sentiment-based Candidate Selection for NMT** (LINK COMING SOON), co-written by me (Alex Jones) and my supervisor [Derry Wijaya](https://derrywijaya.github.io/web/). The paper describes a decoder-side approach for selecting the translation candidate that best preserves the automatically-socred sentiment of the source text. To this end, we train three distinct sentiment classifiers: an English BERT model, a Spanish XLM-RoBERTa model, and an XLM-RoBERTa model fine-tuned on English but used for sentiment classification in other languages, such as French, Finnish, and Indonesian. We compute a softmax over the logits returned by these classifiers to obtain the probability of a text <img src="https://latex.codecogs.com/svg.image?t" title="t" /> being in the positive class, and call this number the "sentiment score" <img src="https://latex.codecogs.com/svg.image?S(t)" title="S(t)" />:

<img src="https://latex.codecogs.com/svg.image?S(t)&space;=&space;P(c=1&space;\mid&space;t)" title="S(t) = P(c=1 \mid t)" />
