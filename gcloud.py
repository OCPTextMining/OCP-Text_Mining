from google.cloud import storage
from config import Config
import glob
import progressbar


def download_raw_files(local_path):
    try:
        storage_client = storage.Client.from_service_account_json(Config.GCLOUD_CREDENTIALS)
    except FileNotFoundError:
        storage_client = storage.Client()
    bucket = storage_client.get_bucket("ocp-raw-data")
    blobs = bucket.list_blobs(prefix='raw-data/')
    for blob in blobs:
        if '.pdf' in blob.name:
            filename = blob.name.split('/')[-1]
            blob.download_to_filename(local_path + filename)


def download_from_cloud_storage(local_path, cloud_folder, limit=None):
    try:
        storage_client = storage.Client.from_service_account_json(Config.GCLOUD_CREDENTIALS)
    except FileNotFoundError:
        storage_client = storage.Client()
    bucket = storage_client.get_bucket("ocp-raw-data")
    blobs = bucket.list_blobs(prefix=cloud_folder+'/')
    for i, blob in enumerate(blobs):
        filename = blob.name.split('/')[-1]
        blob.download_to_filename(local_path + filename)
        if i == limit-1:
            break


def upload_to_cloud_storage(local_folder, cloud_folder, verbose=False):
    try:
        storage_client = storage.Client.from_service_account_json(Config.GCLOUD_CREDENTIALS)
    except FileNotFoundError:
        storage_client = storage.Client()
    bucket = storage_client.get_bucket("ocp-raw-data")
    files = glob.glob(local_folder)

    if not verbose:
        bar = progressbar.ProgressBar(maxval=len(files),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    for i, file in enumerate(files):
        filename = file.split('/')[-1]
        blob = bucket.blob(cloud_folder+filename)
        blob.upload_from_filename(file)
        if verbose:
            print(filename)
        else:
            bar.update(i+1)
    if not verbose:
        bar.finish()
