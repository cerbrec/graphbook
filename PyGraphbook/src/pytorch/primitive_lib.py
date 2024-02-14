import torch
import numpy as np
import re
import os

""" 
Instructions:
All functions use PyTorch tensors unless they are text (in which case they use numpy array)
All inputs and outputs are either scalar or tensor array
"""


def generate_uniform_random_number(shape, random_seed, left_limit, right_limit):
    """ Generate uniform random number with given shape and limits
        If left_limit is decimal, then output is decimal
        If left_limit is integer, then output is integer
    """
    torch.manual_seed(random_seed)
    if isinstance(left_limit, int):
        return torch.randint(left_limit, right_limit, shape)
    if isinstance(left_limit, float):
        return torch.FloatTensor(*shape).uniform_(left_limit, right_limit)


# pylint: disable=W0622
# noinspection PyShadowingBuiltins
def parse_boolean(input):
    """ Parse scalar or array (either pytorch or numpy) into boolean"""
    if isinstance(input, bool):
        return input
    if isinstance(input, np.ndarray):
        return input.astype(bool)
    if isinstance(input, torch.Tensor):
        return input.bool()
    if isinstance(input, np.generic):
        return bool(input)
    if isinstance(input, str):
        return input.lower() == "true"
    if isinstance(input, int):
        return input == 1
    if isinstance(input, float):
        return input == 1.0
    raise ValueError("Cannot parse boolean from input")


def parse_integer(input):
    """ Parse scalar or array into integer"""
    if isinstance(input, int):
        return input
    if isinstance(input, np.ndarray):
        return input.astype(int)
    if isinstance(input, torch.Tensor):
        return input.int()
    if isinstance(input, np.generic):
        return int(input)
    if isinstance(input, str):
        return int(input)
    if isinstance(input, bool):
        return int(input)
    if isinstance(input, float):
        return int(input)
    raise ValueError("Cannot parse integer from input")


def parse_decimal(input):
    """ Parse scalar or array into decimal"""
    if isinstance(input, float):
        return input
    if isinstance(input, np.ndarray):
        return input.astype(float)
    if isinstance(input, torch.Tensor):
        return input.float()
    if isinstance(input, np.generic):
        return float(input)
    if isinstance(input, str):
        return float(input)
    if isinstance(input, bool):
        return float(input)
    if isinstance(input, int):
        return float(input)
    raise ValueError("Cannot parse decimal from input")


def parse_text(input):
    """ Parse scalar or array into text"""
    if isinstance(input, str):
        return input
    if isinstance(input, np.ndarray):
        return input.astype(str)
    if isinstance(input, torch.Tensor):
        return input.numpy().astype(str)
    if isinstance(input, np.generic):
        return str(input)
    if isinstance(input, bool):
        return str(input)
    if isinstance(input, int):
        return str(input)
    if isinstance(input, float):
        return str(input)
    raise ValueError("Cannot parse text from input")


def lower_upper_case_text(text, is_lower_case):
    """ Lower or upper case text or array (numpy) of text"""
    if isinstance(text, str):
        return text.lower() if is_lower_case else text.upper()
    if isinstance(text, np.ndarray):
        return np.char.lower(text) if is_lower_case else np.char.upper(text)
    raise ValueError("Cannot lower or upper case text")


def join_text(text, text_separator, dimension_index, keep_dimension):
    """ Join text or array (numpy) of text"""
    if isinstance(text, str):
        return text
    if isinstance(text, np.ndarray):
        return np.char.join(text_separator, text)
    raise ValueError("Cannot join text")


def index_text_to_one_hot(text, vocabulary):
    """ Convert text or array (numpy) of text to one hot encoding (pytorch tensor) """
    if isinstance(text, str):
        return torch.tensor(vocabulary[text])
    if isinstance(text, np.ndarray):
        return torch.tensor([vocabulary[t] for t in text])
    raise ValueError("Cannot index text to one hot")


def slice_text(text, start_index, end_index):
    """ Slice scalar text from start index to end index """
    if isinstance(text, str):
        return text[start_index:end_index]
    raise ValueError("Cannot slice text")


def insert_text(text, search_regex, inserted_text):
    """ Element-wise insert inserted_text into text based on regular expression search_regex """
    if isinstance(text, str):
        return re.sub(search_regex, inserted_text, text)
    if isinstance(text, np.ndarray):
        return np.char.replace(text, search_regex, inserted_text)
    raise ValueError("Cannot insert text")


def split_text(text, regex, pad_tag):
    """ Split text (or text array) based on regular expression regex, padding with pad_tag
    Example
      text=[["hello world", "hello", "foobar foo bar"]]
      regex=[[""\\s+"", "\\s+"", "\\s+""]]
      pad_tag=[["xxx", "xxx", "xxx"]]
      output = [["hello", "world", "xxx"], ["hello", "xxx", "xxx"], ["foobar", "foo", "bar"]]
    """
    if isinstance(text, str):
        return [t if t else pad_tag for t in re.split(regex, text)]
    if isinstance(text, np.ndarray):
        return np.char.split(text, regex, maxsplit=1)
    raise ValueError("Cannot split text")


def get_text_length(input):
    """ Get length of text (or text array) """
    if isinstance(input, str):
        return len(input)
    if isinstance(input, np.ndarray):
        return np.char.str_len(input)
    raise ValueError("Cannot get text length")


def contain_text(text, search_regex):
    """ Element-wise check whether regular expression in search_regex is contained in text """
    if isinstance(text, str):
        return bool(re.search(search_regex, text))
    if isinstance(text, np.ndarray):
        return np.char.find(text, search_regex) != -1
    raise ValueError("Cannot check whether text contains search regex")


def index_text(text, vocabulary):
    """ Get the index of each element in text against vocabulary

    If an element of text is not found in vocabulary, then it will be given the index equal to the size of vocabulary
    """
    if isinstance(text, str):
        return vocabulary[text] if text in vocabulary else len(vocabulary)
    if isinstance(text, np.ndarray):
        return np.array([vocabulary[t] if t in vocabulary else len(vocabulary) for t in text])
    raise ValueError("Cannot index text")


def replace_text(text, search_regex, replace_with):
    """ Element-wise replace text with replace_with based on regular expression search_regex

    replace_with allows regular expression format specifiers
    """
    if isinstance(text, str):
        return re.sub(search_regex, replace_with, text)
    if isinstance(text, np.ndarray):
        return np.char.replace(text, search_regex, replace_with)
    raise ValueError("Cannot replace text")


def index_text_to_condensed_one_hot(text, vocabulary):
    """ Encode text across its last dimension into condensed one-hot indices against vocabulary

    If an element of text is not found in vocabulary, then it will be given the index equal to the size of vocabulary
    """
    if isinstance(text, str):
        return vocabulary[text] if text in vocabulary else len(vocabulary)
    if isinstance(text, np.ndarray):
        return np.array([vocabulary[t] if t in vocabulary else len(vocabulary) for t in text])
    raise ValueError("Cannot index text to condensed one hot")


def concatenate_text(text_1, text_2):
    """ Element-wise concatenate text_1 and text_2 """
    if isinstance(text_1, str):
        return text_1 + text_2
    if isinstance(text_1, np.ndarray):
        return np.char.add(text_1, text_2)
    raise ValueError("Cannot concatenate text")


def dropout(array, probability):
    """ Randomly zero some of the elements of array based on the corresponding probability """
    return torch.dropout(array, probability, True)


def argument_minimum(array, dimension_index, keep_dimension):
    """ Return the indices of the minimum values of array across dimension_index

    If keep_dimension is true, then argument will keep the dimension at `dimension_index
    """

    return torch.argmin(array, dimension_index, keep_dimension)


def argument_maximum(array, dimension_index, keep_dimension):
    """ Return the indices of the maximum values of array across dimension_index

    If keep_dimension is true, then argument will keep the dimension at `dimension_index
    """

    return torch.argmax(array, dimension_index, keep_dimension)


def divide(left_operand, right_operand):
    """ Element-wise divide left_operand by right_operand """
    return torch.div(left_operand, right_operand)


def error_function(input):
    """Compute the Gauss error function of each element in input"""
    return torch.erf(input)


def subtract(left_operand, right_operand):
    """ Element-wise subtract right_operand from left_operand """
    return torch.sub(left_operand, right_operand)


def element_wise_multiply(left_operand, right_operand):
    """ Element-wise multiply left_operand by right_operand """
    return torch.mul(left_operand, right_operand)


def top_k(array, dimension_index, k):
    """ Return k largest elements and indices from array across dimension_index"""

    return torch.topk(array, k, dimension_index)


def minimum(array, dimension_index, keep_dimension):
    """ Return the minimum values of array across dimension_index

    If keep_dimension is true, then minimum will keep the dimension at `dimension_index
    """

    return torch.min(array, dimension_index, keep_dimension)


def absolute_value(target):
    """ Compute the absolute value of each element in target """
    return torch.abs(target)


# pylint: disable=W0622
# noinspection PyShadowingBuiltins
def sum(array, dimension_index, keep_dimension):
    """ Return the sum values of array across dimension_index

    If keep_dimension is true, then output will keep the dimension at dimension_index
    """

    return torch.sum(array, dimension_index, keep_dimension)


def add(left_operand, right_operand):
    """ Add left_operand and right_operand."""
    return torch.add(left_operand, right_operand)


def argument_multinomial(probabilities, dimension_index, number_samples, random_seed):
    """ Draw samples from a multinomial distribution using probabilities with random_seed across dimension_index

    Return number_samples sampled indices of probabilities
    """

    torch.manual_seed(random_seed)
    return torch.multinomial(probabilities, number_samples, True)


def count_occurrences(target, count_against):
    """ Count and return the occurrences of each element of target in count_against. """

    return torch.bincount(target, count_against)


def maximum(array, dimension_index, keep_dimension):
    """ Return the maximum values of array across dimension_index

    If keep_dimension is true, then maximum will keep the dimension at `dimension_index
    """

    return torch.max(array, dimension_index, keep_dimension)


def element_wise_exponentiate(base, exponent):
    """ Calculate the element-wise exponential of base to the power of exponent"""
    return torch.pow(base, exponent)


def element_wise_logarithm(base, argument):
    """ Calculate the element-wise logarithm of argument to the base """
    return torch.log(argument) / torch.log(base)


def mean(array, dimension_index, keep_dimension):
    """ Return the mean values of array across dimension_index

    If keep_dimension is true, then mean will keep the dimension at `dimension_index
    """

    return torch.mean(array, dimension_index, keep_dimension)


def modulo(left_operand, right_operand):
    """ Element-wise modulo left_operand by right_operand """
    return torch.fmod(left_operand, right_operand)


def logical_and(condition_1, condition_2):
    """ Element-wise calculate logical AND of condition_1 and condition_2 """
    return torch.logical_and(condition_1, condition_2)


def logical_or(condition_1, condition_2):
    """ Element-wise calculate logical OR of condition_1 and condition_2 """
    return torch.logical_or(condition_1, condition_2)


def equal(left_operand, right_operand):
    """ Element-wise check whether left_operand is equal to right_operand """
    return torch.eq(left_operand, right_operand)


def greater_than(left_operand, right_operand):
    """ Element-wise check whether left_operand is greater than right_operand """
    return torch.gt(left_operand, right_operand)


def logical_not(condition):
    """ Element-wise calculate logical NOT of condition """
    return torch.logical_not(condition)


def less_than(left_operand, right_operand):
    """ Element-wise check whether left_operand is less than right_operand """
    return torch.lt(left_operand, right_operand)


def contain(target, check_against):
    """ Check whether each element of target is in check_against """
    return torch.isin(target, check_against)


def conditional_filter(condition, data_if_true, data_if_false):
    """ Filter and merge elements of data_if_true and data_if_false based on boolean condition """

    return torch.where(condition, data_if_true, data_if_false)


def identity(input):
    """ Return input """
    return input


def sample_by_step(array: torch.Tensor, dimension_index, step_value):
    """ Sequentially sample pytorch tensor using step_value across dimension_index

    Example 1:
        array=[[0,1,2,3,4,5,6,7,8,9,10]]
        dimension_index=0
        step_value=2
        output=[0,2,4,6,8,10]

    Example 2:
        array=[[0,1,2,3,4],[5,6,7,8,9]]
        dimension_index=1
        step_value=2
        output=[[0,2,4],[5,7,9]]

    Example 3:
        array=[[[0,1,2,3,4,5,6,7,8]],[[9,10,11,12,13,14,15,16,17]]]
        dimension_index=2
        step_value=3
        output=[[[0,3,6]],[[9,12,15]]]
    """
    # Handle all 3 examples
    if dimension_index == 0:
        return array[::step_value]
    if dimension_index == 1:
        return array[:, ::step_value]
    if dimension_index == 2:
        return array[:, :, ::step_value]
    if dimension_index == 3:
        return array[:, :, :, ::step_value]
    raise ValueError("Cannot sample by step")


def reverse_order(array, dimension_index):
    """ Reverse the order of array across dimension_index """
    return torch.flip(array, [dimension_index])


def filter(array, dimension_index, condition):
    """ Filter array based on condition across dimension_index

    array=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dimension_index=0
    condition=[true, false, true, false, true, true, true, true, false, true]
    filtered_array=[1, 3, 5, 6, 7, 8, 10]

    array=[[0.1, 2.3, 3.4, 4.5, 5.6], [6.7, 7.8, 8.9, 9.0, 10.1]]
    dimension_index=1
    condition=[false, true, false, true, false]
    filtered_array=[[2.3, 4.5], [7.8, 9.0]]

    """
    # Handle examples for dimension_index 1 through 4
    if dimension_index == 0:
        return array[condition]
    if dimension_index == 1:
        return array[:, condition]
    if dimension_index == 2:
        return array[:, :, condition]
    if dimension_index == 3:
        return array[:, :, :, condition]

    raise ValueError("Cannot filter")


def swap_dimensions(array, dimension_index_1, dimension_index_2):
    """ Swap two dimensions (dimension_index_1 and dimension_index_2) of array

    array=[[1, 2, 3, 4], [5, 6, 7, 8]]
    dimension_index_1=0
    dimension_index_2=1
    swapped_array=[[1, 5], [2, 6], [3, 7], [4, 8]]

    array=[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
    dimension_index_1=1
    dimension_index_2=2
    swapped_array=[[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]], [[0.7, 0.9, 1.1], [0.8, 1.0, 1.2]]]

    array=[[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]]]
    dimension_index_1=0
    dimension_index_2=3
    swapped_array=[[[[0.1], [0.3]], [[0.5], [0.7]], [[0.9], [1.1]]], [[[0.2], [0.4]], [[0.6], [0.8]], [[1.0], [1.2]]]]
    """

    # Handle examples for dimension_index 1 through 4
    if dimension_index_1 == 0 and dimension_index_2 == 1:
        return array.transpose(0, 1)
    if dimension_index_1 == 0 and dimension_index_2 == 2:
        return array.transpose(0, 2)
    if dimension_index_1 == 0 and dimension_index_2 == 3:
        return array.transpose(0, 3)
    if dimension_index_1 == 1 and dimension_index_2 == 2:
        return array.transpose(1, 2)
    if dimension_index_1 == 1 and dimension_index_2 == 3:
        return array.transpose(1, 3)
    if dimension_index_1 == 2 and dimension_index_2 == 3:
        return array.transpose(2, 3)

    raise ValueError("Cannot swap dimensions")


def reshape(input, new_shape):
    """ Reshape input into output with the same data and number of elements
    input=[1, 2, 3, 4, 5]
    new_shape=[5, 1]
    output=[[1], [2], [3], [4], [5]]

    input=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    new_shape=[1, 2, 3]
    output=[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]

    input=[[[true, false, true, true], [false, true, false, false], [false, true, false, true]]]
    new_shape=[3, 2, 2]
    output=[[[true, false], [true, true]], [[false, true], [false, false]], [[false, true], [false, true]]]

    input=[[["foo", "bar", "foo1"], ["bar1", "foo2", "bar2"]], [["foo3", "bar3", "foo4"], ["bar4", "foo5", "bar5"]]]
    new_shape=[6, 2]
    output=[["foo", "bar"], ["foo1", "bar1"], ["foo2", "bar2"], ["foo3", "bar3"], ["foo4", "bar4"], ["foo5", "bar5"]]
    """

    # If Input is numpy array, then it is text
    if isinstance(input, np.ndarray):
        return np.reshape(input, new_shape)

    return torch.reshape(input, new_shape)


def get_dimension_size(array, dimension_index):
    """ Get the size of dimension_index of array """
    return array.shape[dimension_index]


def get_shape(input):
    """ Get the shape (i.e. size of each dimension) of input. If numpy array, then it's text. If scalar, shape is empty torch array
    input=[1, 2, 3, 4, 5]
    shape=[5]

    input=[[0.1, 2.3, 4.5], [6.7, 8.9, 10.0]]
    shape=[2, 3]

    input=[[[true, false, true], [true, false, true]], [[false, true, false], [false, true, false]]]
    shape=[2, 2, 3]
    """

    return input.shape



def range(start_value, end_end_value, step_value):
    """ Return a one-dimensional vector with values from start_value (inclusive) to end_value (exclusive)

    Use step_value as the distance between each adjacent point

    start_value=0
    end_value=100
    step_value=5
    vector=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    start_value=-20.5
    end_value=20.5
    step_value=2.5
    vector=[-20.5, -18.0, -15.5, -13.0, -10.5, -8.0, -5.5, -3.0, -0.5, 2.0, 4.5, 7.0, 9.5, 12.0, 14.5, 17.0, 19.5]
    """

    return torch.arange(start_value, end_end_value, step_value)


def get_sub_arrays(input, dimension_index, selected_indices):
    """ Get sub-arrays from input across dimension_index selecting each index of selected_indices

    input=[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500], [1000, 2000, 3000, 4000, 5000], [11, 22, 33, 44, 55], [110, 220, 330, 440, 550], [1100, 2200, 3300, 4400, 5500], [111000, 222000, 333000, 444000, 555000]]
    dimension_index=0
    selected_indices=[3, 4, 6]
    output=[[1000, 2000, 3000, 4000, 5000], [11, 22, 33, 44, 55], [1100, 2200, 3300, 4400, 5500]]
    """

    # Handle dimension indices 1 through 4
    if dimension_index == 0:
        return input[selected_indices]
    if dimension_index == 1:
        return input[:, selected_indices]
    if dimension_index == 2:
        return input[:, :, selected_indices]
    if dimension_index == 3:
        return input[:, :, :, selected_indices]

    raise ValueError("Cannot get sub arrays")


def slice(array, dimension_index, start_index, end_index):
    """ Slice array from start_index (inclusive) to end_index (exclusive) across dimension_index

    array=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dimension_index=0
    start_index=2
    end_index=8
    sliced_array=[3, 4, 5, 6, 7, 8]

    array=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    dimension_index=1
    start_index=1
    end_index=4
    sliced_array=[[2, 3, 4], [8, 9, 10]]

    array=[[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1, 1.2]]]
    dimension_index=2
    start_index=1
    end_index=3
    sliced_array=[[[0.2, 0.3]], [[0.6, 0.7]], [[1.0, 1.1]]]

    array=[[["foo", "bar", "foo1", "bar1", "foo2", "bar2"]], [["foo3", "bar3", "foo4", "bar4", "foo5", "bar5"]]]
    dimension_index=2
    start_index=1
    end_index=5
    sliced_array=[[["bar", "foo1", "bar1", "foo2"]], [["bar3", "foo4", "bar4", "foo5"]]]
    """

    # Handle dimension indices 1 through 4
    if dimension_index == 0:
        return array[start_index:end_index]

    if dimension_index == 1:
        return array[:, start_index:end_index]

    if dimension_index == 2:
        return array[:, :, start_index:end_index]

    if dimension_index == 3:
        return array[:, :, :, start_index:end_index]

    raise ValueError("Cannot slice array")


def concatenate(array_1, array_2, dimension_index):
    """ Concatenate array_1 and array_2 on dimension_index

    array_1=[1, 2, 3, 4, 5]
    array_2=[1, 2, 3, 4, 5]
    dimension_index=0
    concatenated_array=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    array_1=[[true, true], [false, true]]
    array_2=[[true, false], [false, false]]
    dimension_index=1
    concatenated_array=[[true, true, true, false], [false, true, false, false]]

    array_1=[["foo", "bar", "foo_bar", "bar_foo"]]
    array_2=[["foo1", "bar1", "foo2", "bar2"]]
    dimension_index=0
    concatenated_array=[["foo", "bar", "foo_bar", "bar_foo"], ["foo1", "bar1", "foo2", "bar2"]]
    """

    # Handle dimension indices 1 through 4
    if isinstance(array_1, np.ndarray):
        return np.concatenate((array_1, array_2), dimension_index)

    return torch.cat((array_1, array_2), dimension_index)


def deduplicate_to_vector(array):
    """ Deduplicate array into a one-dimensional deduped_vector with all the unique elements

    array=[11, 11, 5, 7, 11, 9, 9, 9, 5, 35, 27]
    deduped_vector=[27, 35, 9, 7, 5, 11]

    array=[[0.1, 0.1, 2.2, 3.5, 3.5], [3.5, 8.9, 8.9, 0.1, 2.2]]
    deduped_vector=[2.2, 8.9, 3.5, 0.1]

    array=[["foo", "foo", "foo1", "bar", "bar"], ["foobar", "bar1", "foo", "foo", "barbar"]]
    deduped_vector=["foo1", "bar1", "barbar", "bar", "foobar", "foo"]
    """

    # Handle numpy array
    if isinstance(array, np.ndarray):
        return np.unique(array)

    return torch.unique(array)


def split(array, dimension_index, element_index):
    """ Split array at element_index across dimension_index

    split_array_left excludes data at element_index and split_array_right includes data at element_index

    array=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dimension_index=0
    element_index=6
    split_array_left=[0, 5, 10, 15, 20, 25]
    split_array_right=[30, 35, 40, 45, 50]

    array=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
    dimension_index=1
    element_index=2
    split_array_left=[[0.1, 0.2], [0.5, 0.6], [0.9, 1.0]]
    split_array_right=[[0.3, 0.4], [0.7, 0.8], [1.1, 1.2]]

    array=[[[true, true, true, true, false, false]], [[true, true, false, true, false, false]]]
    dimension_index=2
    element_index=1
    split_array_left=[[[true]], [[true]]]
    split_array_right=[[[true, true, true, false, false]], [[true, false, true, false, false]]]
    """

    # Handle dimension indices 1 through 4
    if dimension_index == 0:
        return array[:element_index], array[element_index:]

    if dimension_index == 1:
        return array[:, :element_index], array[:, element_index:]

    if dimension_index == 2:
        return array[:, :, :element_index], array[:, :, element_index:]

    if dimension_index == 3:
        return array[:, :, :, :element_index], array[:, :, :, element_index:]

    raise ValueError("Cannot split array")


def broadcast_to_shape(target, shape):
    """ Broadcast each dimension of target to the corresponding dimension of shape

    target=[[1, 2, 3, 4, 5]]
    shape=[2, 5]
    result=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    target=[[1.0, 1.2, 3.3, 6.4, 9.05]]
    shape=[3, 5]
    result=[[1.0, 1.2, 3.3, 6.4, 9.05], [1.0, 1.2, 3.3, 6.4, 9.05], [1.0, 1.2, 3.3, 6.4, 9.05]]
    """
    if isinstance(target, np.ndarray):
        return np.broadcast_to(target, shape)

    return target.expand(shape)


def reduce_one_dimension(input, dimension_index, selected_index):
    """ Reduce input by one dimension choosing selected_index across dimension_index

    input=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    dimension_index=0
    selected_index=2
    output=[5, 6]

    input=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    dimension_index=1
    selected_index=2
    output=[0.3, 0.6, 0.9]

    input=[[[true, true], [false, false]], [[true, false], [false, true]]]
    dimension_index=2
    selected_index=1
    output=[[true, false], [false, true]]

    input=[[["foo", "bar", "foo1", "bar1"]], [["foo2", "bar2", "foo3", "bar3"]], [["foo4", "bar4", "foo5", "bar5"]]]
    dimension_index=2
    selected_index=2
    output=[["foo1"], ["foo3"], ["foo5"]]
    """

    # Handle dimension indices 1 through 4
    if dimension_index == 0:
        return input[selected_index]

    if dimension_index == 1:
        return input[:, selected_index]

    if dimension_index == 2:
        return input[:, :, selected_index]

    if dimension_index == 3:
        return input[:, :, :, selected_index]

    raise ValueError("Cannot reduce one dimension")


def reduce_one_dimension_with_element_wise_selected_indices(array, dimension_index, selected_indices):
    """ Reduce array by one dimension choosing element-wise selected_indices across dimension_index

    array=[1, 2, 3, 4, 5, 6, 7]
    dimension_index=0
    selected_indices=3
    elements=4

    array=[[10, 20, 30], [1, 2, 3]]
    dimension_index=1
    selected_indices=[0, 1]
    elements=[10, 2]

    array=[[[10, 20, 30], [40, 50, 60]], [[1, 2, 3], [4, 5, 6]]]
    dimension_index=0
    selected_indices=[[0, 0, 0], [1, 1, 1]]
    elements=[[10, 20, 30], [4, 5, 6]]
    """

    # Handle dimension indices 1 through 4
    if dimension_index == 0:
        return array[selected_indices]

    if dimension_index == 1:
        return array[:, selected_indices]

    if dimension_index == 2:
        return array[:, :, selected_indices]

    if dimension_index == 3:
        return array[:, :, :, selected_indices]

    raise ValueError("Cannot reduce one dimension with element-wise selected indices")


def expand_one_dimension(input, dimension_index):
    """ Add a new dimension to input at dimension_index and push the original dimension_index to the next dimension

    input=5
    dimension_index=0
    output=[5]

    input=[0.1, 2.3, 4.5, 6.7, 8.9, 10.0]
    dimension_index=1
    output=[[0.1], [2.3], [4.5], [6.7], [8.9], [10.0]]

    input=[[true, false, true, false], [true, true, false, true]]
    dimension_index=2
    output=[[[true], [false], [true], [false]], [[true], [true], [false], [true]]]
    """

    # Handle dimension indices 1 through 4
    return input.unsqueeze(dimension_index)


