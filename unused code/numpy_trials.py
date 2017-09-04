#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
import numpy as np

a = np.array([
              [0,0,0,1,0,0,0],
              [0,0,1,0,1,0,0],
              [0,1,0,0,0,1,0],
              [1,0,0,0,0,0,1]])
section = a[:, 1]
rev_section = section[::-1]
print section, rev_section
max = np.argmax(a[:, :1])
print max