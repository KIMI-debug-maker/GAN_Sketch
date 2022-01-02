import os 
for root,dirs,files in os.walk("./"): 
    
    for file in files: 
        path=os.path.join(root,file)
        f_path=os.path.join('../faker',root[2:],file)
        if file[-2:]=='py':
            print(path)
            with open(path, 'r') as f:
                data = f.read() 
            with open(f_path,'w') as f:
                f.write(data.replace('jt','jt'))

