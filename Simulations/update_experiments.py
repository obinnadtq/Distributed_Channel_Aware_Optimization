import sys
import os
import fileinput
path = os.getcwd()

def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp,replaceExp)
        sys.stdout.write(line)

#----------------------------Path and name--------------------------------------
configuration = 'Configuration7'

exp = 'Exp1'

#-------------------------------------------------------------------------------
exp_path = path + os.sep + configuration + os.sep + exp + os.sep + 'Exp_files'

if not os.path.exists(exp_path):
    print('Exp file directory does not exist. Please create it before using this script')
    sys.exit()

exp_list = os.listdir(exp_path)
os.chdir(exp_path)

#-------------------------------------------------------------------------------

# strings to be updated formed as tuples aranged in a list
update_str = [('chain_optimization = True',''),('aib','sib')]


#-------------------------------------------------------------------------------

for exp in exp_list:

    for str_idx in range(0, len(update_str)):
        replaceAll(exp_path + os.sep + exp , update_str[str_idx][0], update_str[str_idx][1])

