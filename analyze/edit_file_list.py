import os,sys

def add_string_to_lines(file_path, string_to_add, output_file=None):
    """
    Adds a given string to the end of every line in a file.

    :param file_path: Path to the input file.
    :param string_to_add: String to append to each line.
    :param output_file: Path to save the modified content (optional). 
                        If None, it overwrites the original file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = [string_to_add + line for line in lines]

    output_path = output_file if output_file else file_path
    with open(output_path, 'w') as file:
        file.writelines(modified_lines)

# Example usage:
# add_string_to_lines('example.txt', ' - added text')


add_string_to_lines('data/vista/train/image_list.txt', 'train/', 'data/vista/train/image_list_updated.txt')
add_string_to_lines('data/vista/validation/image_list.txt', 'validation/', 'data/vista/validation/image_list_updated.txt')
#
add_string_to_lines('data/domainnet/painting.txt', 'painting/', 'data/domainnet/painting_updated.txt')
add_string_to_lines('data/domainnet/real.txt', 'real/', 'data/domainnet/real_updated.txt')
add_string_to_lines('data/domainnet/sketch.txt', 'sketch/', 'data/domainnet/sketch_updated.txt')
