
"""
Found in: https://python-xmp-toolkit.readthedocs.io/en/latest/using.html
"""

from libxmp import XMPFiles

from os import listdir
from os.path import isfile, join

def copy_xmp_files(original_path, destination_path):

    xmpfile = XMPFiles(file_path=original_path, open_forupdate = True)
    if xmpfile is not None:
        xmp = xmpfile.get_xmp()

    xmpfile2 = XMPFiles(file_path = destination_path, open_forupdate = True)
    
    xmpfile2.put_xmp(xmp) 
    xmpfile.close_file()
    xmpfile2.close_file()


mypath_in = "/home/asoria/Documents/Blurring_andreas/understand-ai-anonymizer/images/ID823311/"
mypath_out = "/home/asoria/Documents/Blurring_andreas/understand-ai-anonymizer/results/ID823311/"

onlyfiles_in = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

for f in onlyfiles_in:
    print(f)
    filename_in = join(mypath_in, f)
    filename_out = join(mypath_out, f)
    copy_xmp_files(filename_in, filename_out)
    
