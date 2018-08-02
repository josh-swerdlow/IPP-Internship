import numpy as np

# inpt = np.array(1.234)
# inpt = np.array([1, 2])
# inpt = np.array([5, 4, 3, 2, 1])
inpt = np.array([[9, 7, 8], [6, 5, 4], [3, 2, 1]])

shape = list(inpt.shape)
while inpt.ndim < 2:
    shape.append(1)
    inpt = np.expand_dims(inpt, 0)
    print(inpt)

cols = shape[0]
rows = shape[1]


print("Input value: {}\n".format(inpt))
print("Rows: {}, Columns: {}\n".format(rows, cols))

empty = ""
for row in range(rows):
    row_str = str()
    for col in range(cols):
        row_str = row_str + "{:.4f} ".format(inpt[row, col])
    row_str = row_str[:-1] + "\n"
    print("Row string: " + row_str)
    empty = empty + row_str

print(empty)
