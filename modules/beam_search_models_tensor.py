import numpy as np
import pandas as pd
import torch


class SubgroupDetails:
    def __init__(self, conjunction, interestingness=None, target_share=None, coverage=None, 
                    no_sg_p_inst=None, no_sg_inst=None):
        self.conjunction = conjunction
        self.interestingness = interestingness 
        self.target_share = target_share
        self.coverage = coverage
        self.no_sg_p_inst = no_sg_p_inst
        self.no_sg_inst = no_sg_inst

    # Here we are only talking about ourselves, the other could be any object so just leave it as is
    # So if we do equivalency between object A and object B. Object A will reference its 
    # __eq__ to define itself and object B will reference its __eq__ to define itself
    def __eq__(self, other):
        return repr(self.conjunction) == repr(other)


    def __repr__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        attrs = sorted(str(sel) for sel in self.conjunction.selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets,))

class Conjunction:

    def __init__(self, selectors):
        self.selectors = selectors

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    def covers(self, data, device):

        # empty description ==> return a list of all '1's
        if not self.selectors:
            return torch.full(len(data), True, dtype=bool)

        # non-empty description, returns a logical vector where all entries that equal to true
        # correspond to rows in the dataset that are covered by the selectors in self.selectors

        combined_tensor = torch.ones(1, len(data), device=device)

        for sel in self.selectors:
            pseudo_logic_vector = sel.covers(data)
            combined_tensor *= pseudo_logic_vector
        return combined_tensor


    # All sorted so for equality of two subgroups the order of selectors in the pattern does not matter
    # 15 == 0 AND 17 ==1 is equivalent to 17 == 1 and 15 == 0
    def __repr__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        attrs = sorted(str(sel) for sel in self.selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets))

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __len__(self):
        return len(self.selectors)



class EquitySelector():

    def __init__(self, attribute, value):

        # Okay so we have the column and the value we expect for the column
        self.attribute = attribute
        self.value = value

    # This defines how comparing instances of this class are computed. 
    # So if you go == between two instances of this class, all your really doing is checking
    # if the repr of the two classes is the same
    def __eq__(self, other):
        return repr(self) == repr(other)

    # Defines how the selectors are hashed when they are inserted into dictionaries? Or when the
    # hash function is called on them
    def __hash__(self):
        return hash(repr(self))


    def covers(self, data):
 
        # Okay so wait this returns a logical vector. Where each entry for the vector corresponds to 
        # Whether this selector is met in that row of the dataset.
        column_data = data[:, self.attribute]
        relu = torch.nn.ReLU()

        # Generating a pseudo logic vector such depending on the binned value
        if self.value.item() == -1:
            column_data = torch.mul(column_data, -1)
            column_data = relu(column_data)
        
        elif self.value.item() == 0:
            column_data = torch.square(column_data)
            column_data = torch.mul(column_data, -2)
            column_data = torch.add(column_data, 1)
            column_data = relu(column_data)

        elif self.value.item() == 1:
            column_data = relu(column_data)

        else:
            raise Exception("Strange binned value occured, not -1, 0 or 1, execution terminated")


        return column_data


    # So this method defines what we see if this object is printed
    def __repr__(self):

        query=""

        query = str(self.attribute) + "==" + str(self.value.item())

        return query    


    # This defines the less than operation: so if we compare 2 instances of this class x < y, then
    # this method is invoked 
    def __lt__(self, other):

        # This repr function generally returns a printable representation of the input, and if the __repr__
        # method is defined then this function accesses this defintion.
        return repr(self) < repr(other)
