from editor import summaryFileEditor
from editor import inputFileEditor

import numpy as np


def main(fn):
    """
    MOOO
    """

    input1 = 5
    input2 = [3.0, 4.1, 4.4, 10.2]
    input3 = "interp"
    input4 = np.array([4.2, 5.1, 5.93])
    inputs = [input1, input2, input3, input4]

    inpt_editor = inputFileEditor(inpt_fn="test_inpt", inputs=inputs)
    sum_editor = summaryFileEditor("test_sum.json")

    inpt_editor.attributes()
    print("\n")
    sum_editor.attributes()

    inpt_editor.write()
    sum_editor.write(inpt_editor)

    test_inptFile = sum_editor.get()

    print(test_inptFile)





if __name__ == '__main__':
    main("param/op12a_171122022_FeLBO3")

