from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from datasets import ClassLabel
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoTokenizer
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')


class TestModel:

    def __init__(self, model, token):
        self.model_name = model
        self.model = None
        self.tokeinzer = None
        self.token=token
        self.datasets = ["SetFit/CR", "SetFit/sst2", "SetFit/sst5", 'stanfordnlp/imdb',
                         "SetFit/20_newsgroups", "SetFit/enron_spam", "AmazonScience/massive",
                         'PolyAI/banking77', 'takala/financial_phrasebank','ag_news', 'dair-ai/emotion',
                         "MoritzLaurer/cap_sotu", 'cornell-movie-review-data/rotten_tomatoes']
        self.pipeline = None

        self.macro_scores = []
    def load_model(self):
        self.model = GLiClassModel.from_pretrained(self.model_name, token=self.token).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token, add_prefix_space=True)
        self.pipeline = ZeroShotClassificationPipeline(self.model, self.tokenizer, classification_type='single-label',
                                                       device='cuda:0')

    def prepare_dataset(self, dataset, classes=None, text_column='text', label_column="label_text", split=None):

        if 'test' in dataset:
            test_dataset = dataset['test']
        elif isinstance(dataset, Dataset):
            test_dataset = dataset
        else:
            test_dataset = dataset['train']
        if classes is None:
            classes = test_dataset[label_column]
            classes = list(set(classes))
            if split is not None:
                classes = [' '.join(class_.split(split)) for class_ in classes]
        texts = test_dataset[text_column]
        true_labels = test_dataset[label_column]
        print(true_labels[:5])
        print(classes)
        if type(test_dataset[label_column][0]) == int:
            true_labels = [classes[label] for label in true_labels]
        return texts, classes, true_labels

    def prepare_nomapping(self, dataset, classes=None, text_column='text', label_column='label_text', split=None):
        if 'test' in dataset:
            test_dataset = dataset['test']
        elif isinstance(dataset, Dataset):
            test_dataset = dataset
        else:
            test_dataset = dataset['train']
        if classes is None:
            if isinstance(test_dataset.features[label_column], ClassLabel):
                classes = test_dataset.features[label_column].names
            else:
                classes = test_dataset[label_column]
                classes = list(set(classes))
                if split is not None:
                    classes = [' '.join(class_.split(split)) for class_ in classes]
        texts = test_dataset[text_column]
        true_labels = test_dataset[label_column]
        # if isinstance(test_dataset.features[label_column], ClassLabel):
        #     true_labels = [test_dataset.features[label_column].int2str(label) for label in true_labels]
        if type(true_labels[0]) == int:
            true_labels = [classes[label] for label in true_labels]

        return texts, classes, true_labels

    def get_gliclass_predictions(self, test_texts, classes, batch_size=8):
        results = self.pipeline(test_texts, classes, batch_size=batch_size)
        predicts = [result[0]['label'] for result in results]
        return predicts

    def evaluate(self, predicts, true_labels):
        micro = f1_score(true_labels, predicts, average="micro")
        macro = f1_score(true_labels, predicts, average="macro")
        weighted = f1_score(true_labels, predicts, average="weighted")
        return {"micro": micro, "macro": macro, "weighted": weighted}

    def process(self):
        self.load_model()
        for dataset in self.datasets:
            classes = None
            print(dataset)
            if dataset == 'SetFit/sst5':
                classes = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, classes=classes)
            elif dataset == 'PolyAI/banking77':
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='text', label_column='label')
            elif dataset == 'takala/financial_phrasebank':
                ds = load_dataset('takala/financial_phrasebank', 'sentences_allagree', trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='sentence',
                                                                          label_column="label")
            elif dataset == "AmazonScience/massive":
                ds = load_dataset(dataset,"en-US")
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='utt',
                                                                          label_column="intent")
            elif dataset == 'stanfordnlp/imdb':
                ds = load_dataset(dataset, trust_remote_code=True)
                classes = ['negative', 'positive']
                test_texts, classes, true_labels = self.prepare_nomapping(ds, classes=classes, text_column='text', label_column='label')
                print(true_labels[0], classes)
            elif dataset == 'ag_news':
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='text', label_column='label')
            elif dataset == 'dair-ai/emotion':
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='text', label_column='label')
            elif dataset == 'MoritzLaurer/cap_sotu':
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='text', label_column='labels')
            elif dataset == 'cornell-movie-review-data/rotten_tomatoes':
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='text', label_column='label')
            elif dataset == 'massive':
                ds = load_dataset("AmazonScience/massive", "en-US", trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds, text_column='utt', label_column='intent')
            else:
                ds = load_dataset(dataset, trust_remote_code=True)
                test_texts, classes, true_labels = self.prepare_nomapping(ds)
            predicts = self.get_gliclass_predictions(test_texts, classes, batch_size=8)
            results = self.evaluate(predicts, true_labels)
            self.macro_scores.append(results['macro'])
            print(results)
        print('Average Score:', np.mean(self.macro_scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TestModel with arguments")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--api_key", type=str, required=False, default = None, help="API key for authentication")

    args = parser.parse_args()

    gliclasstest = TestModel(args.model, args.api_key)
    gliclasstest.process()