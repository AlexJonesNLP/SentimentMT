# SentimentMT

This is the repo associated with the paper **Sentiment-based Candidate Selection for NMT** (LINK COMING SOON), co-written by me (Alex Jones) and my supervisor [Derry Wijaya](https://derrywijaya.github.io/web/). The paper describes a decoder-side approach for selecting the translation candidate that best preserves the automatically-socred sentiment of the source text. To this end, we train three distinct sentiment classifiers: an English BERT model, a Spanish XLM-RoBERTa model, and an XLM-RoBERTa model fine-tuned on English but used for sentiment classification in other languages, such as French, Finnish, and Indonesian. We compute a softmax over the logits returned by these classifiers to obtain the probability of a text <img src="https://render.githubusercontent.com/render/math?math=t"> being in the positive class, and call this number the "sentiment score" <img src="http://www.sciweavers.org/tex2img.php?eq=%24%24S%28t%29%24%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="$$S(t)$$" width="36" height="18" />:

<img src="http://www.sciweavers.org/tex2img.php?eq=%24%24S%28t%29%20%3D%20P%28c%3D1%7Ct%29%24%24&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="$$S(t) = P(c=1|t)$$" width="136" height="18" /> 
