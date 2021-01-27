import argparse
import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd
import torch
from nltk import tokenize
from transformers import BertTokenizerFast

@dataclass
class ExtractedMention(object):
	sentence: Optional[str]
	sent_idx: Optional[int]
	start_word_idx: int
	end_word_idx: int
	mention: str
	tag: str


class Predictor(object):

	MAX_LEN = 256
	BATCH_LEN = 16

	def __init__(self, torch_model_file, biobert_pretrained_model_dir, idx2tag_file_path):
		self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.torch_model = torch.load(torch_model_file, map_location=self.torch_device)
		self.torch_model.eval()
		self.bert_tokenizer = BertTokenizerFast.from_pretrained(biobert_pretrained_model_dir, do_lower_case = False)

		with open(idx2tag_file_path, 'r') as f:
			self.idx2tag = json.loads(f.read())
			self.tag2idx = {v:int(k) for k, v in self.idx2tag.items()}

	def get_sentence_mappings(self, sentences):
		"""
		Computes and returns the following maps from a list of sentences:
			sent_token_idx2word: map (sent_idx, word_idx) -> word_tokens
			sent_word_idx2token: map (sent_idx, token_idx) -> word
		The maps are used after prediction, to map tokens back into their corresponding
		words, and compute tags at the word, rather than token level
		"""
		sent_token_idx2word = {}
		sent_word_idx2token = {}

		for sent_idx, sentence in enumerate(sentences):
			words = tokenize.word_tokenize(sentence)
			curr_token_idx = 0
			for word_idx, word in enumerate(words):
				word_tokens = self.bert_tokenizer.tokenize(word)
				sent_word_idx2token[(sent_idx, word_idx)] = word_tokens
				for i, token in enumerate(word_tokens):
					sent_token_idx2word[(sent_idx, curr_token_idx + i)] = (token, word_idx, word)
				curr_token_idx += len(word_tokens)

		return sent_token_idx2word, sent_word_idx2token

	def tokenize_and_predict(self, sentences):
		"""
		Tokenizes a list of sentences and makes predictions
		Returns list of predictions, per token (this includes 'O' and 'PAD' tags)
		Prediction code runs faster on cuda
		"""
		if not sentences:
			return []

		num_batches = int(len(sentences) / self.BATCH_LEN) + 1
		predictions = []
		self.torch_model.to(self.torch_device)
		for i in range(num_batches):
			batch_sentences = sentences[i * self.BATCH_LEN : (i + 1) * self.BATCH_LEN]
			if len(batch_sentences) > 0:
				encoded_inputs = self.bert_tokenizer(batch_sentences, return_tensors='pt', padding = True, truncation = True, max_length = self.MAX_LEN)
				batch_inputs = encoded_inputs['input_ids'].to(self.torch_device)
				batch_attention_mask = encoded_inputs['attention_mask'].to(self.torch_device)
				with torch.no_grad():
					batch_logits = self.torch_model(batch_inputs, attention_mask = batch_attention_mask)[0]
					batch_preds = torch.argmax(batch_logits, dim = 2).cpu()
					for sent_preds, sent_attention_mask in zip(batch_preds, batch_attention_mask):
						preds = sent_preds[sent_attention_mask != 0][1:-1]
						predictions.extend([preds])
		return predictions

	def get_mentions_and_spans(self, predictions, sent_token_idx2word):
		"""
		Returns list of mentions of interest (B-MET, I-MET, B-DAT)
		annotations: map (sent_idx, word_idx, word) -> list of token tags for that word
		"""
		annotations = {}
		for sent_idx, sent_pred_tags in enumerate(predictions):
			for tag in ['B-MET', 'I-MET', 'B-DAT']:
				# Note: Use -1 default value, in case we're using a dev time-only model produced from a small training
				# dataset that is missing one of the above tags
				tag_idx = self.tag2idx.get(tag, -1)
				mentions = torch.where(sent_pred_tags == tag_idx)[0].cpu().numpy()
				if len(mentions) > 0:
					for token_idx in mentions:
						(token, word_idx, word) = sent_token_idx2word[(sent_idx, token_idx)]
						if (sent_idx, word_idx, word) in annotations:
							annotations[(sent_idx, word_idx, word)] += [tag]
						else:
							annotations[(sent_idx, word_idx, word)] = [tag]
		return annotations

	def get_word_tags(self, annotations, sent_word_idx2token):
		"""
		Gets word-level tokens; A word gets a 'B-MET', 'I-MET', 'B-DAT' tag if 
		all of its tokens have that tag as well; else, it gets 'O'
		filtered_annotations: list (sent_idx, word_idx, word, word_tag)
		"""
		filtered_annotations = []
		for (sent_idx, word_idx, word), tags in annotations.items():
			counter = Counter(tags)
			most_common_tag = counter.most_common(1)[0][0]
			num_most_common_tag = counter.most_common(1)[0][1]
			num_word_tokens = len(sent_word_idx2token[(sent_idx, word_idx)])
			if num_word_tokens == num_most_common_tag:
				label = most_common_tag
			else:
				label = 'O'
			filtered_annotations.append((sent_idx, word_idx, word, label))
		return filtered_annotations

	def get_dataframe(self, mentions: List[ExtractedMention]):
		"""
		Adds annotations in a pd dataframe format
		"""
		df = pd.DataFrame([asdict(m) for m in mentions])
		df['tag'] = df['tag'].replace('I-MET', 'MET')
		df['tag'] = df['tag'].replace('B-MET', 'MET')
		df['tag'] = df['tag'].replace('B-DAT', 'DAT')
		return df

	def merge_word_annotations(self, annotations, text_sentences) -> pd.DataFrame:
		"""
		Handles merging consecutive word-level tags part of the same mention,
		in one mention
		"""
		mentions = []
		start_word_idx = -1
		end_word_idx = -1
		last_tag = None
		last_sent_idx = -1
		last_sentence = ''
		running_mention = ''
		start_new_mention = False

		for annotation in annotations:
			curr_sent_idx, curr_word_idx, curr_word, curr_tag = annotation
			curr_sentence = text_sentences[curr_sent_idx]
			if curr_word in string.punctuation:
				continue
			# words are part of the same mention and are being merged
			for tag in ['MET', 'DAT']:
				if curr_tag == 'B-' + tag:
					# append last mention
					if start_new_mention:
						mentions.append(ExtractedMention(
								sent_idx=last_sent_idx,
								sentence=last_sentence,
								start_word_idx=start_word_idx,
								end_word_idx=end_word_idx,
								mention=running_mention,
								tag=last_tag
						))
					# start new mention
					start_new_mention = True
					running_mention = curr_word
					start_word_idx = curr_word_idx
					end_word_idx = curr_word_idx
					last_sent_idx = curr_sent_idx
					last_tag = curr_tag
					last_sentence = curr_sentence
				elif curr_tag == 'I-' + tag:
					if (curr_sent_idx == last_sent_idx) and (curr_word_idx == end_word_idx + 1):
						running_mention += ' ' + curr_word
						end_word_idx = curr_word_idx
		if start_new_mention:
			mentions.append(ExtractedMention(
				sent_idx=last_sent_idx,
				sentence=last_sentence,
				start_word_idx=start_word_idx,
				end_word_idx=end_word_idx,
				mention=running_mention,
				tag=last_tag
			))
			return self.get_dataframe(mentions)
		else:
			return pd.DataFrame()

	def predict(self, text: str) -> pd.DataFrame:
		"""
		:return: pd.DataFrame with ExtractedMention attributes as columns
		"""
		print('Tokenizing text...')
		text_sentences = tokenize.sent_tokenize(text)
		sent_token_idx2word, sent_word_idx2token = self.get_sentence_mappings(text_sentences)

		print(f'Sentences={len(text_sentences)}, ' 
					f'Max sentence length={max([len(s) for s in text_sentences] or [0])}, '
					f'Sentence lengths > {self.MAX_LEN}: {[len(s) > self.MAX_LEN for s in text_sentences].count(True)}')

		print('Making predictions...')
		predictions = self.tokenize_and_predict(text_sentences)
		print('Get mentions from predictions...')
		annotations = self.get_mentions_and_spans(predictions, sent_token_idx2word)
		print('Get word tags...')
		filtered_annotations = self.get_word_tags(annotations, sent_word_idx2token)
		print('Merge word annotations...')

		final_annotations_df = self.merge_word_annotations(filtered_annotations, text_sentences)
		print(f'Predicted {final_annotations_df.shape[0]} annotation(s) for {len(text_sentences)} sentences')
		return final_annotations_df

def main() -> None:
	default_test_input_filename = 'papers/example.txt'
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-file-name', '-i', default=default_test_input_filename)
	parser.add_argument('--output-file-name', '-o', type=str, default=None)
	args = parser.parse_args()
	print(f'Predicting for file {args.input_file_name}')
	with open(args.input_file_name, mode='r', encoding='utf-8') as f:
		text = f.read()
	predictor = Predictor(
		'model_artifacts/model.pt',
		'model_artifacts/biobert_vocab.txt',
		'model_artifacts/idx2tag.json'
	)
	result = predictor.predict(text)

	if args.output_file_name:
		with open(args.output_file_name, 'w') as out:
			out.write(result.to_csv())
	else:
		print(result.to_csv())
        
if __name__ == '__main__':
	main()