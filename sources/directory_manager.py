# -*- coding: utf-8 -*-


import os

class DirectoryManager(): 
    
    def __init__(self):
        self.cwd = os.getcwd()
    
    
    def make_multiple_directories(self, path_list, option='replace'):        
        for path in path_list:
            self.make_directory(path, option)

    def make_directory(self, path, option='replace'): # option: replace(remove all)/pass
        fullpath = os.path.join(self.cwd, path)
        existence = os.path.exists(path)
        if not existence:
            os.mkdir(path)
            print('{}  WAS CREATED'.format(fullpath))        
        else:
            if option == 'replace':
                self.clean_directory(path)
                print('{}  WAS REPLACED'.format(fullpath))
            elif option == 'pass':
                print('{}  ALREADY EXISTED'.format(fullpath))

#    def check_elements(self, path):
#        all_elements = os.walk(path)
#        for root, dirs, files in all_elements:
#            print("path：", root)
#            print("directory：", dirs)
#            print("file：", files)
    
    def clean_directory(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
                
