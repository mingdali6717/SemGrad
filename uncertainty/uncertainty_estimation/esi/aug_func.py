import math
import random
import copy
import re
from itertools import combinations
import numpy as np
import nlpaug.augmenter.char as aug_char
import nlpaug.augmenter.word as aug_word

def sub_chars(prompts, augnum=5, percent=0.5):
    """
    change the letter case of random selected chars
    prompts: List[str]
    """
    subed_prompts = []
    for p in prompts:
        indices = [i for i, char in enumerate(p) if char.isalpha()]
        num_to_changes = math.ceil(percent * len(indices))
        if len(indices) <= 10:
            
            all_sub_indexes = list(combinations(indices, num_to_changes))
            if augnum > len(all_sub_indexes):
                print("WARNING: augmentation number is larger than avaliable numbers, some augmented sentences will be repetative.")
                p_add_augnum =  augnum - len(all_sub_indexes)
                p_augnum = len(all_sub_indexes)
            else:
                p_add_augnum = 0
                p_augnum = augnum

            change_indexes = random.sample(all_sub_indexes, p_augnum)
            if p_add_augnum > 0:
                for _ in range(p_add_augnum):
                    change_indexes.extend(random.sample(all_sub_indexes, 1))
        else:
            change_indexes = []
            for _ in range(augnum):
                try_num = 0
                while try_num <= 5:
                    idx = random.sample(indices, num_to_changes)
                    if idx in change_indexes:
                        try_num += 1
                    else:
                        break
                change_indexes.append(idx)

        subed_prompts.append([char_case_sub(p, idx) for idx in change_indexes])
        

        
    return subed_prompts



def sub_whole_word_chars(prompts, augnum=5, percent=0.5):
    """
    change the whole word to uppercase if most of its chars are lowercase, otherwise to lowercase
    """
    def contains_word(string):
        return bool(re.search("[A-Za-z]", string))
    subed_prompts = []
    for p in prompts:
        words = p.split(" ")
        
        indices = [i for i, w in enumerate(words) if contains_word(w)]
        num_to_changes = math.ceil(percent * len(indices))
        if len(indices) <= 10:
            all_sub_indexes = list(combinations(indices, num_to_changes))
            if augnum > len(all_sub_indexes):
                print("WARNING: augmentation number is larger than avaliable numbers, some augmented sentences will be repetative.")
                p_add_augnum =  augnum - len(all_sub_indexes)
                p_augnum = len(all_sub_indexes)
            else:
                p_add_augnum = 0
                p_augnum = augnum

            change_indexes = random.sample(all_sub_indexes, p_augnum)
            if p_add_augnum > 0:
                for _ in range(p_add_augnum):
                    change_indexes.extend(random.sample(all_sub_indexes, 1))
        else:
            change_indexes = []
            for _ in range(augnum):
                try_num = 0
                while try_num <= 5:
                    idx = random.sample(indices, num_to_changes)
                    if idx in change_indexes:
                        try_num += 1
                    else:

                        break
                change_indexes.append(idx)
        subed_prompts.append([" ".join(word_case_sub(words, idx)) for idx in change_indexes])
    
    return subed_prompts

def cap_first_char(prompts, augnum=5, percent=0.5):
    def contains_word(string):
        return bool(re.search("[A-Za-z]", string))
    subed_prompts = []
    for p in prompts:
        words = p.split(" ")
        
        indices = [i for i, w in enumerate(words) if contains_word(w)]
        num_to_changes = math.ceil(percent * len(indices))
        if len(indices) <= 10:
            all_sub_indexes = list(combinations(indices, num_to_changes))
            if augnum > len(all_sub_indexes):
                print("WARNING: augmentation number is larger than avaliable numbers, some augmented sentences will be repetative.")
                p_add_augnum =  augnum - len(all_sub_indexes)
                p_augnum = len(all_sub_indexes)
            else:
                p_add_augnum = 0
                p_augnum = augnum

            change_indexes = random.sample(all_sub_indexes, p_augnum)
            if p_add_augnum > 0:
                for _ in range(p_add_augnum):
                    change_indexes.extend(random.sample(all_sub_indexes, 1))
        else:
            change_indexes = []
            for _ in range(augnum):
                try_num = 0
                while try_num <= 5:
                    idx = random.sample(indices, num_to_changes)
                    if idx in change_indexes:
                        try_num += 1
                    else:

                        break
                change_indexes.append(idx)
        subed_prompts.append([" ".join(_cap_first_char(words, idx)) for idx in change_indexes])
    
    return subed_prompts

def skip_one_char(prompts, augnum=5, percent=0.5, min_char_num=3):
    def contains_word(string):
        return bool(re.search("[A-Za-z]", string))
    subed_prompts = []
    for p in prompts:
        words = p.split(" ")
        
        indices = [i for i, w in enumerate(words) if contains_word(w) and len(w)> min_char_num]
        num_to_changes = math.ceil(percent * len(indices))
        if len(indices) <= 10:
            all_sub_indexes = list(combinations(indices, num_to_changes))
            if augnum > len(all_sub_indexes):
                p_add_augnum =  augnum - len(all_sub_indexes)
                p_augnum = len(all_sub_indexes)
            else:
                p_add_augnum = 0
                p_augnum = augnum

            change_indexes = random.sample(all_sub_indexes, p_augnum)
            aug_ps = [" ".join(_skip_one_char(words, idx, min_char_num=min_char_num)) for idx in change_indexes]
            if p_add_augnum > 0:
                for _ in range(p_add_augnum):
                    try_num = 0
                    while try_num < augnum+5:
                        idx = random.sample(indices, num_to_changes)
                        aug_p = " ".join(_skip_one_char(words, idx, min_char_num=min_char_num))
                        if aug_p not in aug_ps:
                            break
                        else:
                            try_num += 1
                    if try_num == augnum:
                        print(f"there is repetition augmention for prompt '{p}'")
                    aug_ps.append(aug_p)   
            subed_prompts.append(aug_ps)
                
        else:
            change_indexes = []
            for _ in range(augnum):
                try_num = 0
                while try_num <= augnum+5:
                    idx = random.sample(indices, num_to_changes)
                    if idx in change_indexes:
                        try_num += 1
                    else:

                        break
                change_indexes.append(idx)
            subed_prompts.append([" ".join(_skip_one_char(words, idx, min_char_num=min_char_num)) for idx in change_indexes])
    
    return subed_prompts


def sub_keyboard_typo(prompts, augnum=5, percent=0.5):
    
    def contains_word(string):
        return bool(re.search("[A-Za-z]", string))
    subed_prompts = []
    for p in prompts:
        words = p.split(" ")
        
        indices = np.array([i for i, w in enumerate(words) if contains_word(w) and len(w)>=4])
        num_to_changes = math.ceil(percent * len(indices))
        weights = np.array([i + 1 for i in range(len(indices))])
        aug_ps = []
        for _ in range(augnum):
            try_num = 0
            while try_num < 2 * augnum:
                if try_num < augnum:
                    idx = np.random.choice(indices, size=num_to_changes, p=weights/weights.sum(), replace=False).tolist()
                else:
                    idx = np.random.choice(indices, size=num_to_changes, replace=False).tolist()
                aug_p = " ".join(_sub_keyboard_typo(words, idx))
                if aug_p not in aug_ps:
                    break
                else:
                    try_num += 1
            if try_num == 2 * augnum:
                print(f"there is repetition augmention for prompt '{p}'")
            aug_ps.append(aug_p)    
        subed_prompts.append(aug_ps)
    return subed_prompts


def word_aug(prompts, aug_method="spelling", augnum=5, percent=0.3, ):
    if aug_method == "spelling":

        aug_func = aug_word.SpellingAug(aug_min=1, aug_p=percent)
    elif aug_method == "synonym":
        aug_func = aug_word.synonym.SynonymAug(aug_min=1, aug_p=percent)
    elif aug_method == "del":
        aug_func = aug_word.random.RandomWordAug(aug_min=1, aug_p=percent)
    elif aug_method == "swap":
        aug_func = aug_word.random.RandomWordAug(action="swap", aug_min=1, aug_p=percent)
    elif aug_method == "antonym":
        aug_func = aug_word.antonym.AntonymAug(aug_min=1, aug_max=1)
    subed_prompts = []
    for p in prompts:
        aug_ps = aug_func.augment(p, n=augnum)
        
        if len(list(set(aug_ps))) < len(aug_ps):
            print(f"there is repetition augmention for prompt '{p}'")

        subed_prompts.append(aug_ps)
    return subed_prompts

def sub_spelling_typo(prompts, augnum=5, percent=0.3):
    return word_aug(prompts, aug_method="spelling", augnum=augnum, percent=percent)

def sub_synonym(prompts, augnum=5, percent=0.3):
    return word_aug(prompts, aug_method="synonym", augnum=augnum, percent=percent)
def sub_antonym(prompts, augnum=5, percent=0.3):
    return word_aug(prompts, aug_method="antonym", augnum=augnum)

def del_words(prompts, augnum=5, percent=0.3):
    return word_aug(prompts, aug_method="del", augnum=augnum, percent=0.1)

def swap_words(prompts, augnum=5, percent=0.3):
    return word_aug(prompts, aug_method="swap", augnum=augnum, percent=percent)



def trivia_sub(prompts, augnum=5):
    return [[s]*augnum for s in prompts]

def char_case_sub(text, change_indices):
    
    chars = list(text)
    for index in change_indices:
        if chars[index].islower():
            chars[index] = chars[index].upper()
        else:
            chars[index] = chars[index].lower()
    return ''.join(chars)

def word_case_sub(word_list, change_indices):
    new_word_list = copy.deepcopy(word_list)

    for idx in change_indices:
        word = word_list[idx]
        lower_num = len([char for char in word if char.isalpha() and char.islower()])
        upper_num = len([char for char in word if char.isalpha() and char.isupper()])
        if lower_num >= upper_num:
            new_word_list[idx] = "".join([c.upper() if (c.isalpha() and c.islower()) else c for c in word])
        else:
            new_word_list[idx] = "".join([c.lower() if (c.isalpha() and c.isupper()) else c for c in word])
    return new_word_list

def _cap_first_char(word_list, change_indices):
    new_word_list = copy.deepcopy(word_list)

    for idx in change_indices:
        word = word_list[idx]
        char_idxs = [(i, char) for i, char in enumerate(word) if char.isalpha()]
        case_list = [1 if c[1].isupper() else 0 for c in char_idxs]
        chars = list(word)
        if len(char_idxs) == 1: # if contain only one char, change its case
            chars[char_idxs[0][0]] = char_idxs[0][1].upper() if char_idxs[0][1].islower() else char_idxs[0][1].lower()
        elif case_list[0] == 1 and sum(case_list[1:]) == 0: # if the the word alrealy captialized it's first letter, lower case its first letter
            chars[char_idxs[0][0]] = char_idxs[0][1].lower()
        else: # captalize first letter and then lowercase all other letters
            chars = [c.lower() if (c.isalpha() and c.isupper()) else c for c in word]
            chars[char_idxs[0][0]] = char_idxs[0][1].upper()
        
        new_word_list[idx] = "".join(chars)

    return new_word_list

def _skip_one_char(word_list, change_indices, min_char_num=3):
    new_word_list = copy.deepcopy(word_list)

    for idx in change_indices:
        word = word_list[idx]
        char_idxs = [(i, char) for i, char in enumerate(word)]
        index_can_be_skipped = [c[0] for c in char_idxs[min_char_num:]] # do not skip first 3 chars
        idx_to_skip = random.sample(index_can_be_skipped, 1)[0]

        chars = [c for i,c in enumerate(word) if i!=idx_to_skip]
        
        
        new_word_list[idx] = "".join(chars)

    return new_word_list

def _add_white_char(word_list, change_indices):
    white_chars = [
        " ",  # Space
        "\t",  # Horizontal Tab
    ]
    new_word_list = copy.deepcopy(word_list)
    for i in change_indices:
        new_word_list[i] = new_word_list[i] + random.choice(white_chars)
    return  new_word_list

def _sub_keyboard_typo(word_list, change_indices):
    aug = aug_char.KeyboardAug(aug_word_min=1, aug_word_p=1, aug_char_max=1,aug_char_min=1, include_special_char=False, include_upper_case=False, include_numeric=False, min_char=1)
    new_word_list = copy.deepcopy(word_list)
    for i in change_indices:
        subword = new_word_list[i][3:]
        
        new_word_list[i] = new_word_list[i][:3] + aug.augment(subword)[0]
    return  new_word_list


