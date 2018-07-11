from editor import inputFileEditor


def main(fn):
    editor = inputFileEditor.load(fn)
    print(editor.inpt_fn, editor.sum_fn)


if __name__ == '__main__':
    main("Test117")
