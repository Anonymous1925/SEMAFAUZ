import argparse
import numpy as np
import torch
import transformers
import datasets
import re
import cuda
import os
import csv
import pickle
import time
import methods
import sememe_coref as sc

parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('model', type=str, help="Path to the model")
parser.add_argument('dataset_path', type=str, help="Hugging face path to dataset")
parser.add_argument('set', type=str, help="Set to train on between train, validation and test")
parser.add_argument('--comment', type=str, default='default', help="Short comment added to the results")
parser.add_argument('--dict_path', default="data/gender/male_female.csv", type=str, help="Path to words file")
parser.add_argument('--length', type=int, default=512, help="The length truncation of the labels used")
parser.add_argument('--depen_ablation', action='store_true')
parser.add_argument('--single_ablation', action='store_true')
parser.add_argument('--checking_ablation', action='store_true')
parser.add_argument('--coref_ablation', action='store_true')
parser.add_argument('--sememe_ablation', action='store_true')
parser.add_argument('--mutation_only', action='store_true')

args = parser.parse_args()
model_path = args.model
dataset_path = args.dataset_path
dict_path = args.dict_path
comment = args.comment
trunc_length = args.length
split_of_dataset = args.set
depen_ablation = args.depen_ablation
single_ablation = args.single_ablation
checking_ablation = args.checking_ablation
coref_ablation = args.coref_ablation
sememe_ablation = args.sememe_ablation
mutation_only = args.mutation_only
full_lists_name, new_word_list_A, new_word_list_B = methods.getWordlist(dict_path)

if coref_ablation :
    depen_ablation=True
    sememe_ablation=True
    single_ablation=True
    checking_ablation=True

job_path = "../data/gender/male_female_job.csv"
job_file_name, job_list_A, job_list_B = methods.getWordlist(job_path)



device = torch.device('cuda') if cuda else torch.device('cpu')
model_config = transformers.PretrainedConfig.from_pretrained(model_path)

model_prediction_type = model_config.problem_type if model_config.problem_type == 'single_label_classification' else "multi_label_classification"
isMulti = model_config.problem_type != 'single_label_classification'

dict_labels = model_config.id2label
num_labels = len(dict_labels)
model_labels = list(dict_labels.values())
model_name = model_config.name_or_path

# Processing Dataset
dataset_name = model_config.finetuning_task
complete_dataset = datasets.load_dataset(dataset_path, name=dataset_name, data_dir='data', split=split_of_dataset)
complete_dataset_size = len(complete_dataset)

# For each case, join the sentences in order to represent the case into one string
dataset_target_column = 'labels' if isMulti else 'label'
dataset_text = [''.join(complete_dataset[i][('text')]).lower() for i in
                range(complete_dataset_size)]  # IN LOWER CASE
dataset_labels = [complete_dataset[i][dataset_target_column] for i in range(complete_dataset_size)]
if not isinstance(dataset_labels[0], list):
    for x in range(complete_dataset_size): dataset_labels[x] = [dataset_labels[x]]
for x in range(complete_dataset_size): dataset_labels[x].sort()

output_path = "output" + "/" + dataset_name + "/" + model_name + "/" + split_of_dataset + "/"
methods.createDir(output_path)

# Loading Tokenizer and Model
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type=model_prediction_type)
model.to(device)

# truncation of text to replace only used tokens
truncated_text = methods.getTruncated(output_path, dataset_text, tokenizer, trunc_length)

model_base_prediction = methods.getPrediction(output_path, truncated_text, tokenizer, model, isMulti, trunc_length,
                                              device)

sc.init_sememe_coref([new_word_list_A, new_word_list_B], [job_list_A, job_list_B])

print("Using model from {}\n defined as {} which name is {}\n with dataset of {} named {}"
      "\n Using {} set of length {}.\nDoing smart replacement.\nModel is of type {} and has {} output Labels."
      "\nWord file used is {} with {} tuples."
      .format(model_path, type(model).__name__, model_name, dataset_path, dataset_name,
              split_of_dataset, complete_dataset_size, model_prediction_type, num_labels, full_lists_name,
              str(len(new_word_list_A))))

start_time = time.time()

def is_Different(array1, array2):
    base = set(array1)
    prediction = set(array2)
    if not base == prediction:
        return True
    return False

def fct_smart_replacement():
    nb_discarded_cases = 0
    nb_mutants = 0
    mutants = []
    nb_errors = 0
    for i in range(len(truncated_text)):
        if i % 10 == 0 :
            print("Doing case {}/{}".format( i, len(truncated_text)))
        case_error = 0
        smart_generation = sc.getSentenceReplacements(truncated_text[i])
        if smart_generation == None :
            nb_discarded_cases += 1
            continue
        nb_mutants += len([1 for x in smart_generation if x[7] != 0])
        for e in smart_generation:
            mutant = e[1]
            predicted_class_id = methods.predict(mutant, tokenizer, model, isMulti, trunc_length,
                                                device)
            increment = 1 if is_Different(model_base_prediction[i], predicted_class_id) and e[7] != 0 else 0
            case_error += increment
            nb_errors += increment
            e += [increment, truncated_text[i], i]
            if case_error > 0:
                print(">>Case {}/{} modified {} time(s) for {} error(s) ".format(i, complete_dataset_size, len(smart_generation),
                                                                           case_error))
        mutants.append(smart_generation)
    print("Total of {} mutants and {} errors.".format(nb_mutants, nb_errors))

    return mutants, nb_mutants, nb_errors, nb_discarded_cases
sc.checking_ablation = checking_ablation
sc.coreference_ablation = coref_ablation
sc.dependency_ablation = depen_ablation
sc.single_ablation = single_ablation
sc.sememe_ablation = sememe_ablation

print("sc.dependency_ablation =",sc.dependency_ablation)
print("sc.single_ablation =",sc.single_ablation)
print("sc.checking_ablation =",sc.checking_ablation)
print("sc.coreference_ablation =",sc.coreference_ablation)
print("sc.sememe_ablation =",sc.sememe_ablation)


mutants, nb_mutants, nb_errors, nb_discarded_cases = fct_smart_replacement()
testing_time = time.time() - start_time
columns = ["Male-Female", "Mutant", "Coref_modification", "Depen_modification", "Atomic_modification"
           , "Nb_dictionary", "Nb_sememe", "Pass_simi_check", "Error", "Original_trunc", "Case_ID"]
print("Testing took {} seconds.".format(testing_time))

# Saving pickle with errors
content = [[columns, nb_mutants, nb_errors, nb_discarded_cases, testing_time,comment], mutants]
pickle_path = output_path + 'error_details' + '/' + comment + "_mutants.pkl"
print("Saving details in {}.".format(pickle_path))
file = methods.createOpen(pickle_path, "wb")
pickle.dump(content, file)
file.close()

#Saving unknown words
dict_path = "../output/dict_unknown.pkl"

print("Saving dict ...")
if os.path.isfile(dict_path):
    dict_from_file = methods.getFromPickle(dict_path, "rb")
    dict_from_file.update(set([(x[0],x[1][2]) for x in sc.dict_unk_words.items() if x[1][0] != sc.NO_GENDER]))
else:
    dict_from_file = set([(x[0],x[1][2]) for x in sc.dict_unk_words.items() if x[1][0] != sc.NO_GENDER])
methods.writePickle(dict_from_file, dict_path, 'wb')
print("exit")
