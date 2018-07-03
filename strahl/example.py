from editor import FileEditor


def main():
    f = FileEditor('test')

    print('--> Current Line Number: {}'.format(f.file.line))
    print(f.file.readline())
    print('--> Current Line Number: {}'.format(f.file.line))
    f.file.reset_position()
    print('--> Current Line Number: {}'.format(f.file.line))
    f.add('\nHello\n')
    print(f.file.readline())
    f.delete()
    print(f.file.readline())


if __name__ == '__main__':
    main()
