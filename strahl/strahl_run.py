import os
import sys
import subprocess as sub
import source.interface.auxillary as aux


def strahl_run(inpt_fns=None, strahl_cmd="./strahl"):
    """
    Runs strahl using an input file to fill in the parameters that have been
    commented out from parameter files.

    Parameters:
        * **inpt_fns** [list/str|None]: name(s) of the input file in a list or string
        * **strahl_cmd** [str|"./strahl"]: the command to run strahl which can
            be replaced with different tags that are described in the user manual.

    """
    if inpt_fns is None:
        aux.print_dirContents(os.curdir)

        prompt = "Enter and input file(s) seperated by space or exit [enter]: "
        inpt_fns = input(prompt)

        if inpt_fns is "":
            sys.exit("Exiting")

    if isinstance(inpt_fns, str):
        inpt_fns = inpt_fns.split()

    if not isinstance(inpt_fns, list):
        inpt_fns = list(inpt_fns)

    for inpt_fn in inpt_fns:
        if not os.path.isfile("./" + inpt_fn):
            print("Error: {} is not a valid file".format(inpt_fn))
            inpt_fn = input("Enter the correct file or exit [enter]: ")

            if inpt_fn is "":
                sys.exit("Exiting")

        sub.call(strahl_cmd.split())


if __name__ == '__main__':
    strahl_run()

