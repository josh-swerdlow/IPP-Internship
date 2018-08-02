# import os
# import re
# import sys
# import auxillary as aux
# import shutil as shell
# from editor import inputFileEditor


# def initializeInputFile(*parameterFiles, inpt_fn=None):
#     if inpt_fn is not None and os.path.isfile(inpt_fn):
#         print("{} is an input file found in {}"
#             .format(aux.colorFile(inpt_fn), aux.colorDir(os.curdir)))

#         new = input("Would you like to overwrite {}, "
#                     .format(aux.colorFile(inpt_fn)) +
#                     "load it into a new file, or exit [Y/N/Enter]? ")

#         if re.match("(y|Y)", new):
#             sum_fn = inputFileEditor.load_sumFile(inpt_fn)

#         elif re.match("(n|N)", new):
#             sum_fn = os.path.join("summaries",
#                         inputFileEditor.load_sumFile(inpt_fn))

#             newInpt_fn = inputFileEditor.create_inptFile()
#             newSum_fn = os.path.join("summaries",
#                         inputFileEditor.create_sumFile(newInpt_fn))

#             shell.copyfile(inpt_fn, newInpt_fn)
#             shell.copyfile(sum_fn, newSum_fn)

#         elif new is "":
#             print("Exiting Program")
#             sys.exit()

#         else:
#             print("Error: incorrect answer given...trying again.\n\n")
#             inputFileEditor.load(inpt_fn)

#     else:
#         print("Error: {} was not found in {}\n"
#             .format(aux.colorFile(inpt_fn),
#                     aux.colorDir("~/" + os.curdir)) +
#             "This is the current directory.\n")

#         aux.print_dirContents(os.curdir)

#         inpt_fn = input("Please enter a different input file, " +
#                 "create a new input file [new file name], " +
#                 "or exit [enter]: ")

#         if inpt_fn is '':
#             print("Exiting program")
#             sys.exit()

#         else:
#             if not os.path.isfile(inpt_fn):
#                 inpt_fn = inputFileEditor.create_inptFile(inpt_fn)
#                 sum_fn = inputFileEditor.create_sumFile(inpt_fn)

#                 print("Creating {} and {} as input and summary files."
#                     .format(aux.colorFile(inpt_fn),
#                         aux.colorPckl(sum_fn)))

#             else:
#                 inputFileEditor.load(inpt_fn, parameterFiles)

#     return inputFileEditor(inpt_fn, sum_fn, parameterFiles)


#     @classmethod
#     def load(cls, inpt_fn=None):  # [check]
#         if inpt_fn is None:
#             aux.print_dirContents(os.curdir)

#             inpt_fn = input("What input file would you like to load? ")

#         if not os.path.isfile(inpt_fn):
#             print("Error: The file {} does not exists."
#                   .format(aux.colorFile(inpt_fn)))

#             aux.print_dirContents(os.curdir)

#             inpt_fn = input("Please enter an input file or exit [enter]: ")

#             if inpt_fn is "":
#                 sys.exit("Exiting")

#             else:
#                 return cls.load(inpt_fn)

#         return cls(inpt_fn)