## create a function to download thee data
import requests
import os

def download_data(url , target_folder , filename = None):

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f'Directory {target_folder} created !!')

    if filename is None :
        filename = url.split('/')[-1]

    file_path = os.path.join(target_folder , filename)

    print(f'[INFO] downlaoding {filename} .....')

    try :
        response = requests.get(url , stream= True)
        response.raise_for_status()

        with open(file_path , 'wb') as f :
            for chunk in response.iter_content(chunk_size=4194304):
                f.write(chunk)

        print(f'Successfully saved in : {file_path}')
        return file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
