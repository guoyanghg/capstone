import os
import zipfile

def unzip():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    zip_path = './stage1_train.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data')

if __name__ == '__main__':
    unzip()