import subprocess
import configargparse
import os 

def download_dataset(scenes):
    link_dict = {
        "coke":"https://www.dropbox.com/scl/fo/wtdj9p6sjjithhulhu8f5/h?rlkey=ogapbo5gwbmgmbd5esvfy4k1i&dl=0", 
        "kennedy":"https://www.dropbox.com/scl/fo/oisjfzwjk5wq31ghs57c5/h?rlkey=g4jal3ic3t26ec6m3mnepvm0x&dl=0",
        "david":"https://www.dropbox.com/scl/fo/20ymoe4yulqcpax114ha2/h?rlkey=9yknawfeuibzy8fuzdc62vrhv&dl=0",
        "diffraction":"https://www.dropbox.com/scl/fo/syn5pqgzlocxjnb4txob4/h?rlkey=flilwpmq2iahb3vqqrzmdfvyc&dl=0", 
        "mirror":"https://www.dropbox.com/scl/fo/fmq0r2gko3f5uu4b8qyhq/h?rlkey=eb3ktzube570ekzz7p4t77ayv&dl=0", 
        "cornell":"https://www.dropbox.com/scl/fo/82k5hnonczndxu8mbwbqa/h?rlkey=n8qdj7oghxoq0yixv6wan34h6&dl=0", 
        "pots":"https://www.dropbox.com/scl/fo/5bie9p710ubtnxypjzafu/h?rlkey=kijxjyimskbzllmfdx32h78px&dl=0", 
        "peppers":"https://www.dropbox.com/scl/fo/4ct2wfhvhfcojnb4oqrhl/h?rlkey=0o8yps2q330qxf8hq5h3gucmo&dl=0", 
        "caustics":"https://www.dropbox.com/scl/fo/kkmdzvpyk6xslbf94x7fa/h?rlkey=v9vkiw9nl8c7luhrutuif5v58&dl=0"
    }
    
    for folder in scenes:
        os.makedirs(f"dataset/{folder}", exist_ok = True)
        os.makedirs(f"dataset/{folder}/training_files", exist_ok = True)

        command = f'wget "{link_dict[folder]}" -O dataset/{folder}/training_files.zip'
        subprocess.run(command, shell=True)
        
        command = f"unzip dataset/{folder}/training_files.zip -d dataset/{folder}/training_files"
        subprocess.run(command, shell=True)

        command = f"rm dataset/{folder}/training_files.zip"
        subprocess.run(command, shell=True)
        

if __name__=="__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', help='list of files to download')
    args = parser.parse_args()
    
    final_scenes = []
    all_scenes = ["coke", "kennedy", "diffraction", "mirror", "david", "cornell", "peppers", "pots", "caustics"]
    
    if "all" in args.scenes:
        final_scenes = all_scenes.copy()
    
    if "captured" in args.scenes:
        final_scenes += ["coke", "kennedy", "diffraction", "mirror", "david"]
    if "simulated" in args.scenes:
        final_scenes += ["cornell", "peppers", "pots", "caustics"]
        
    for scene in all_scenes:
        if scene in args.scenes:
            final_scenes += [scene]
    
    final_scenes = list(set(final_scenes))
    download_dataset(final_scenes)



