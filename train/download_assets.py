from io import BytesIO
from zipfile import ZipFile
import tempfile
import requests
import subprocess

def download_unzip(file_url, dst_path):
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))
    zipfile.extractall(dst_path)
    return dst_path


download_unzip(
    'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip',
    'assets'
)

download_unzip(
    'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
    'assets'
)


subprocess.call(['git', 'clone', 'https://github.com/google-research/bert', os.path.join('assets', 'bert')])
