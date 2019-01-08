# Next Utterance Retrieval Model Inspired by "Training Millions of Personalized Dialogue Agents"
In this project, I implemented next utterance retrieval model inspired by Facebook's paper: [Training Millions of Personalized Dialogue Agents](https://arxiv.org/abs/1809.01984)

## Notes
- This project assumes a task that predicts a utterance (response) from the previous 10 sentences (contexts) and the previous sentence (query).
- For model architecture, I used next utterance retrieval model inspired by [this paper](https://arxiv.org/abs/1809.01984).
<img src="https://github.com/noowad/pytorch-next-utterance-retrieval/blob/master/architecture.png" alt="plot" title="plot">

- In my case, since I used different dataset than the original paper, I replaced "Persona" with contexts and "Context" with query.
- In my case, only one response is inputted, and the model predicts whether the response is suitable for the next sentence or not.
(in original paper, multiple responses are inputted and they use softmax for selection of one best response.)
- I used Mecab for tokenization (My dataset is written in Japanese).
## Requirements
- Python 3
- torch
- numpy
- fire
- six
- MeCab
## Execution
- STEP 0, Prepare the data. (previous 10 utterances for contexts, previous 1 utterance for query and utterance for response)
- STEP 1, Adjust hyper parameters in `config.py`.
- STEP 2, Run `python main.py` for training and evaluation.
## Future works
- performance evaluation
- Fasttext pretrained embedding
## References
- I refered the codes of [https://github.com/dhlee347/pytorchic-bert](https://github.com/dhlee347/pytorchic-bert) for pytorchic coding style.
