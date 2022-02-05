import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizerFast, BertTokenizer, ElectraTokenizer, RobertaTokenizer, AutoTokenizer, BertForTokenClassification, ElectraForTokenClassification, RobertaForTokenClassification, AdamW, get_linear_schedule_with_warmup, get_scheduler
from string import punctuation
import seqeval
from seqeval.metrics import classification_report as seqeval_classif_report


def get_tag2idx(df):
	"""
	Returns tags maps from a given dataframe df
	Outputs:
		tag2idx: map from tag to idx
		idx2tag: map from idx to tag
	"""
	tag_values = list(df["tag"].unique()) + ['PAD']
	tag2idx = {tag:idx for idx, tag in enumerate(tag_values)}
	idx2tag = {idx:tag for tag, idx in tag2idx.items()}
	return tag2idx, idx2tag


def tokenize_sentence_words(tokenizer, sentence_words, sentence_word_tags):
	"""
	Tokenizes each word in a sentence using a given tokenizer (eg: BiobertTokenizer)
	For each word in the sentence, asociates its tag to the corresponding word tokens
	Inputs:
		tokenizer: the tokenizer used (eg: BiobertTokenizer)
		sentence_words: the sentence split into words, as a list
		sentence_word_tags: the tags associated with each of the words in sentence_words
	Outputs:
		sentence_tokenized: the sentence split into tokens, as a list
		sentence_tokenized_tags: the tags associated with each of the tokens in the sentence
	"""
	sentence_tokenized = []
	sentence_tokenized_tags = []

	for word_idx, word in enumerate(sentence_words):
		word_tokens = tokenizer.tokenize(str(word))
		word_tag = sentence_word_tags[word_idx]
		num_word_tokens = len(word_tokens)
		sentence_tokenized.extend(word_tokens)
		sentence_tokenized_tags.extend([word_tag] * num_word_tokens)
	return sentence_tokenized, sentence_tokenized_tags


def tokenize_sentences(tokenizer, df, tags_of_interest):
	"""
	Tokenizes a list of sentences retrieved from a given dataframe df
	Inputs:
		tokenizer: the tokenizer used (eg: BiobertTokenizer)
		df: dataframe containing training data (eg: training_data.csv)
	Outputs:
		sentences_tokenized: list of tokenized sentences
		sentences_tokenized_tags: list of tags associated with each tokenized sentence
	"""
	sentences_tokenized = []
	sentences_tokenized_tags = []

	fn1 = lambda x: [word for word in x["word"].values]
	fn2 = lambda x: [tag for tag in x["tag"].values]
	sentences_words = df.groupby(['pmcid', 'sent_index']).apply(fn1)
	sentences_word_tags = df.groupby(['pmcid', 'sent_index']).apply(fn2)

	for sent_idx, sentence_words in enumerate(sentences_words):
		sentence_word_tags = sentences_word_tags[sent_idx]
		for tag in tags_of_interest:
			if tag in sentence_word_tags:
				sentence_tokenized, sentence_tokenized_tags = tokenize_sentence_words(tokenizer, sentence_words, sentence_word_tags)
				sentences_tokenized.append(sentence_tokenized)
				sentences_tokenized_tags.append(sentence_tokenized_tags)
				break;
	return sentences_tokenized, sentences_tokenized_tags


def pad_sequences(sentences_input_ids, maxlen, pad_value):
	return np.array([np.pad(a[:maxlen], (0, max(0, maxlen - len(a))), constant_values=(pad_value)) for a in sentences_input_ids])

def get_train_data(sentences_tokens, tokenizer, max_len):
	encoded_input_ids_all = []
	attention_masks_all = []
	for sentence_tokens in sentences_tokens:
		sentence_tokens_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
		inputs = tokenizer.prepare_for_model(sentence_tokens_ids, padding = 'max_length', \
                                             truncation = True, max_length = max_len, add_special_tokens=False)
		encoded_input_ids = inputs['input_ids']
		attention_mask = inputs['attention_mask']
		encoded_input_ids_all.append(encoded_input_ids)
		attention_masks_all.append(attention_mask)
	return encoded_input_ids_all, attention_masks_all

def get_train_tags(sentences_tags, max_len, token_to_idx = None, pad_value = 0):
	sentence_tags_all = []
	for sentence_tags in sentences_tags: 
		O_tag_idx = token_to_idx['O']
		sentence_tags_all.append([token_to_idx[token] for token in sentence_tags])
	sentence_tags = pad_sequences(sentence_tags_all, max_len, pad_value)
	return sentence_tags


def get_dataloader(inputs, masks, tags):
	"""
	Returns a DataLoader instance that contains inputs, masks and tags
	The DataLoader is used for batching during training
	"""
    
	inputs = torch.tensor(inputs)
	masks = torch.tensor(masks)
	tags = torch.tensor(tags)
    
	dataset = TensorDataset(inputs, masks, tags)
	sampler = SequentialSampler(dataset)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
	return dataloader
    

def print_metrics_resource_type(classif_report, phase, resource_type):
	"""
	Prints metrics (precision, recall) for one resource type
	Resource type can be one of: 'B-MET', 'I-MET', 'B-DAT', 'O'
	"""
	resource_stats = classif_report[resource_type]
	precision = round(resource_stats['precision'], 3)
	recall = round(resource_stats['recall'], 3)
	if (recall + precision) != 0:        
		f1 = round(2 * recall * precision / (recall + precision), 3)
	else:
		f1 = 0
	if resource_type == 'O':
		print(phase + ':', resource_type, 'Precision:    ', precision, resource_type, 'Recall:    ', recall, 'F1:    ', f1)
	else:
		print(phase + ':', resource_type, 'Precision:', precision, resource_type, 'Recall:', recall, 'F1:', f1)
	return f1


def print_metrics_token_level(classif_report, phase, tags):
	"""
	Prints metrics (precision, recall) for each of the resouce type in a given phase
	Phase can be either 'Train' or 'Val'
	"""
	f1s = []
	for tag in tags:
		f1_tag = print_metrics_resource_type(classif_report, phase, tag)
		f1s.append(f1_tag)
	return np.array(f1s).mean()

def print_metrics_entity_level(classif_report, loss, phase):
    """
    Prints metrics (precision, recall) for each of the resouce type in a given phase at the entity level
    Phase can be either 'Train' or 'Val'
    """
    f1_scores = []
    for res_type in ['MET', 'DAT']:
        resource_stats = classif_report[res_type]
        precision = round(resource_stats['precision'], 3)
        recall = round(resource_stats['recall'], 3)
        f1_score = round(resource_stats['f1-score'], 3)
        f1_scores.append(f1_score)
        print(phase + ':', res_type, 'Precision:', precision, 'Recall:', recall, 'F1:', f1_score)
    return np.mean(f1_scores)
        
def evaluate_entity_level(pred_numeric_tags, true_numeric_tags, idx2tag, loss, phase):
    """
    Evaluates metrics on mentions on a given batch at the entity level
    Inputs:
        pred_numeric_tags: predicted labels for a given batch
        true_numeric_tags: true labels for a given batch
        idx2tag: tag index to tag mapping
        phase: 'Train' or 'Val' - for printing purposes only
    """
    pred_tags = [[idx2tag[pred_numeric_tag] for pred_numeric_tag in pred_numeric_tags]]
    true_tags = [[idx2tag[true_numeric_tag] for true_numeric_tag in true_numeric_tags]]
    classif_report_relaxed = seqeval_classif_report(true_tags, pred_tags, digits=3, output_dict = True)
    f1_relaxed = print_metrics_entity_level(classif_report_relaxed, loss, phase)
    return classif_report_relaxed, f1_relaxed

def train_epoch(model, optimizer, lr_scheduler, dataloader, idx2tag, tags_of_interest, tokens, phase):
	"""
	Handles training of the model for one epoch
	"""
	running_loss = 0
	num_batches = len(dataloader)

	pred_tags = []
	true_tags = []
    
	pad_token_id = tokens['PAD']
	cls_token_id = tokens['CLS']
	sep_token_id = tokens['SEP']

	for num_batch, batch in enumerate(dataloader):
		inputs, masks, tags = batch
		inputs = inputs.to(device)
		masks = masks.to(device)
		tags = tags.to(device)

		if phase == 'Train':
			model.zero_grad()
			outputs = model(inputs, attention_mask=masks, labels=tags)
			loss = outputs[0]
			loss.backward()
			optimizer.step()
			if lr_scheduler:
				lr_scheduler.step()
			optimizer.zero_grad()
		else:
			with torch.no_grad():
			    outputs = model(inputs, attention_mask=masks, labels=tags)

		mask_no_pad_tokens = ((inputs != cls_token_id) & (inputs != pad_token_id) & (inputs != sep_token_id))
		tags_no_pad_tokens = tags[mask_no_pad_tokens]
		logits = outputs[1]
		predictions_no_pad_tokens = torch.argmax(logits, axis = 2)[mask_no_pad_tokens]

		predictions_cpu = predictions_no_pad_tokens.to('cpu').numpy()
		tags_cpu = tags_no_pad_tokens.to('cpu').numpy()
		loss = outputs[0].item()
		running_loss += loss

		true_tags.extend(tags_cpu)
		pred_tags.extend(predictions_cpu)
	classif_report_entity, f1_entity = evaluate_entity_level(pred_tags, true_tags, idx2tag, running_loss, phase)
	return running_loss / num_batches, classif_report_entity, f1_entity


def train_model(model, optimizer, lr_scheduler, num_epochs, train_dataloader, val_dataloader, idx2tag, tags, tokens, model_name, checkpt_dir):
	"""
	Handles training of the model for num_epochs
	"""
	max_f1_val = 0    
	train_losses, val_losses = [], []
	for num_epoch in range(num_epochs):
		print('Epoch: ', (num_epoch + 1))
		print('*' * 20)
		# training mode
		model.train()
		train_loss, classif_report_train, f1_train = train_epoch(model, optimizer, lr_scheduler, train_dataloader, idx2tag, tags, tokens, 'Train')
		train_losses.append(train_loss)

		# evaluation mode
		model.eval()
		val_loss, classif_report_val, f1_val = train_epoch(model, None, lr_scheduler, val_dataloader, idx2tag, tags, tokens, 'Val')
		val_losses.append(val_loss)
		print('Train Loss: ', round(train_loss, 5), 'Val Loss: ', round(val_loss, 5))
		if f1_val >= max_f1_val:
			max_f1_val = f1_val
			best_epoch = num_epoch
			best_model = model
	torch.save({
	'model_state_dict': best_model.state_dict(),
	'epoch' : best_epoch,
	'f1_val' : max_f1_val,
	}, os.path.join(checkpt_dir, '/checkpt_' + model_name + '_' + str(best_epoch + 1) + '_epochs'))
	return train_losses, val_losses, best_model


def save_idx2tag(idx2tag, output_data_dir):
	"""
	Save idx2tag dict to json file to dir specified via `--output_data_dir` option. This dict data is used during
	prediction to interpret model output.
	:param idx2tag: The dict
	:return: the file name written
	"""
	print(f'idx2tag={idx2tag}')
	csv_file_name = os.path.join(output_data_dir, 'idx2tag-925.json')
	with open(csv_file_name, 'w') as f:
		f.write(json.dumps(idx2tag))
	return csv_file_name

def get_dataloaders(tokenizer, train_df, val_df, test_df, tags, tag2idx, max_len):
    """
    Returns train, val and test dataloaders from the corresponding dfs
    """
    train_sentences_tokenized, train_sentences_tokenized_tags = tokenize_sentences(tokenizer, train_df, tags)
    train_input_ids, train_attention_masks = get_train_data(train_sentences_tokenized, tokenizer, max_len)
    train_tags = get_train_tags(train_sentences_tokenized_tags, max_len, tag2idx, tag2idx['PAD'])

    val_sentences_tokenized, val_sentences_tokenized_tags = tokenize_sentences(tokenizer, val_df, tags)
    val_input_ids, val_attention_masks = get_train_data(val_sentences_tokenized, tokenizer, max_len)
    val_tags = get_train_tags(val_sentences_tokenized_tags, max_len, tag2idx, tag2idx['PAD'])

    test_sentences_tokenized, test_sentences_tokenized_tags = tokenize_sentences(tokenizer, test_df, tags)
    test_input_ids, test_attention_masks = get_train_data(test_sentences_tokenized, tokenizer, max_len)
    test_tags = get_train_tags(test_sentences_tokenized_tags, max_len, tag2idx, tag2idx['PAD'])


    train_dataloader = get_dataloader(train_input_ids, train_attention_masks, train_tags)
    val_dataloader = get_dataloader(val_input_ids, val_attention_masks, val_tags)
    test_dataloader = get_dataloader(test_input_ids, test_attention_masks, test_tags)
    return train_dataloader, val_dataloader, test_dataloader

def create_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	print("Directory '% s' created" % dir_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# hyperparameters sent by the client are passed as command-line arguments to the script.
	parser.add_argument('--epochs', help = 'number of epochs for training', type=int, default=10)
	parser.add_argument('--batch-size', help = 'batch_size for training', type=int, default=32)
	parser.add_argument('--max_len', help = 'maximum length of sequences in batch', type=int, default=256)
	parser.add_argument('--learning-rate', help = 'learning Rate for training', type=float, default=3e-5)
	parser.add_argument('--weight_decay', help = 'weight decay for learning rate',  type=float, default=0.0)
	parser.add_argument('--use-cuda', help = 'if to use cuda for training or not', type=bool, default=True)
	parser.add_argument('--tags', type=str, help = 'tags present in the dataset', default=['B-DAT', 'B-MET', 'I-MET'])
	parser.add_argument('--data-dir', help = 'directory where the data is being read from', type=str, default='data')
	parser.add_argument('--checkpt_dir', help = 'output directory where checkpoints will be saved', type=str, default='checkpts')
	parser.add_argument('--intermediate-files-dir', help = 'output directory where intermediate files like idx2tag.json will be saved', type=str, default='model_artifacts')
	parser.add_argument('--model_name', help = 'name of model that will be instantiated during training; code currently supports: biobert, scibert, pubmedbert, pubmedbert_pmc, bluebert, bluebert_mimic3, sapbert, \
		sapbert_mean_token, bioelectra, bioelectra_pmc, electra_med, biomed_roberta, biomed_robera_chemprot, biomed_roberta_rct500', type=str, default='scibert')
	parser.add_argument('--model_version', help = 'Huggingface version of model; for instance, for scibert it could be allenai/scibert_scivocab_uncased', type=str, default = 'allenai/scibert_scivocab_uncased')
	parser.add_argument('--train_file', help = 'location of training file', type=str, default='train_v0.csv')
	parser.add_argument('--val_file', help = 'location of val file', type=str, default='val_v0.csv')
	parser.add_argument('--test_file', help = 'location of test file', type=str, default='test_v0.csv')
	parser.add_argument('--sanity_check', help = 'true for sanity checking that the pipeline works well; will only train on 100 entries from the training file', default = False, action = 'store_true', required = False)

	args, _ = parser.parse_known_args()

	data_dir = args.data_dir
	train_df = pd.read_csv(os.path.join(data_dir, args.train_file))
	test_df = pd.read_csv(os.path.join(data_dir, args.test_file))
	val_df = pd.read_csv(os.path.join(data_dir, args.val_file))
	create_dir(args.checkpt_dir)
	create_dir(args.intermediate_files_dir)

	if args.sanity_check:
		train_df = train_df[:10000]

	model_mappings = {
		'biobert' : 'dmis-lab/biobert-v1.1',
		'scibert' : 'allenai/scibert_scivocab_uncased',
		'pubmedbert' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
		'pubmedbert_pmc' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
		'bluebert' : 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
		'bluebert_mimic3' : 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
		'sapbert' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
		'sapbert_mean_token' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token',
		'bioelectra' : 'kamalkraj/bioelectra-base-discriminator-pubmed',
		'bioelectra_pmc' : 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
		'electramed' : 'giacomomiolo/electramed_base_scivocab_1M', 
		'biomed_roberta' : 'allenai/biomed_roberta_base',
		'biomed_robera_chemprot' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169',
		'biomed_roberta_rct500' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500'
	}

	model_name = args.model_name
	model_version = model_mappings[model_name]

	print(f'args={args}')
	print('Loading training data ...')
	print('=' * 30)
	device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
	tag2idx, idx2tag = get_tag2idx(train_df)
	save_idx2tag(idx2tag, args.intermediate_files_dir)

	print('Loading BertTokenizer and model ...', model_version)
	print('=' * 30)
	if 'electra' in model_name:
	    # Example for a BERT-based model. All possible variations are available in the README
	    tokenizer = ElectraTokenizer.from_pretrained(model_version)
	    model = ElectraForTokenClassification.from_pretrained(model_version,
	                                                       num_labels=len(tag2idx),
	                                                       output_attentions=False,                                                       output_hidden_states=False)
	elif 'roberta' in model_name:
	    # Example for an ELECTRA-based model. All possible variations are available in the README
	    tokenizer = AutoTokenizer.from_pretrained(model_version)
	    model = RobertaForTokenClassification.from_pretrained(model_version,
	                                                       num_labels=len(tag2idx),
	                                                       output_attentions=False,
	                                                       output_hidden_states=False)
	elif 'bert' in model_name:
	    # Example for an Roberta-based model. All possible variations are available in the README
	    tokenizer = AutoTokenizer.from_pretrained(model_version)
	    model = BertForTokenClassification.from_pretrained(model_version,
	                                                       num_labels=len(tag2idx),
	                                                       output_attentions=False,
	                                                       output_hidden_states=False)

	model.to(device)
	print('Finished loading BertTokenizer and model!')
	print()

	train_dataloader, val_dataloader, test_dataloader = get_dataloaders(tokenizer, train_df, val_df, test_df, args.tags, tag2idx, args.max_len)
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	num_epochs = args.epochs
	num_training_steps = num_epochs * len(train_dataloader)
	use_lr_scheduler = False

	if use_lr_scheduler:
	    lr_scheduler = get_scheduler(
	        "linear",
	        optimizer = optimizer,
	        num_warmup_steps = int(0.06 * len(train_dataloader)),
	        num_training_steps = num_training_steps
	    )
	else:
	    lr_scheduler = None

	if 'roberta' in model_name:
	    tokens = {'PAD' : tokenizer.vocab["<pad>"], 
	              'SEP' : tokenizer.vocab["</s>"], 
	              'CLS' : tokenizer.vocab["<s>"]}
	elif 'electra' in model_name or 'bert' in model_name:
	    tokens = {'PAD' : tokenizer.vocab["[PAD]"], 
	              'SEP' : tokenizer.vocab["[SEP]"], 
	              'CLS' : tokenizer.vocab["[CLS]"]}

	print('Starting model training ...')
	print('=' * 30)
	train_losses, val_losses, best_model = train_model(model, optimizer, lr_scheduler, 10, train_dataloader, val_dataloader, idx2tag, args.tags, tokens, model_name, args.checkpt_dir)
	print('Finished model training!')
	test_loss, classif_report_test, f1_test = train_epoch(best_model, None, None, test_dataloader, idx2tag, args.tags, tokens, 'Test')
