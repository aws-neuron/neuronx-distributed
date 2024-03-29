import os
import torch
import boto3
import shutil
import botocore
from io import BytesIO
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseCheckpointStorage:

    def __init__(self, dirname: str):
        self._dirname = dirname

    def dirname(self):
        return self._dirname

    @abstractmethod
    def dir_exists(self, dirname: str):
        raise NotImplementedError

    @abstractmethod
    def file_exists(self, filename: str):
        raise NotImplementedError

    @abstractmethod
    def is_checkpoint_xser(self, checkpoint_dir):
        raise NotImplementedError

    @abstractmethod
    def list_checkpoint_tags(self, checkpoint_dir: str):
        raise NotImplementedError

    @abstractmethod
    def save_text(self, text: str, filename: str):
        raise NotImplementedError

    @abstractmethod
    def save_object(self, obj: object, filename: str):
        raise NotImplementedError

    @abstractmethod
    def load_object(self, filename: str, map_location=None) -> object:
        raise NotImplementedError

    @abstractmethod
    def create_dir(self, dirname: str, exist_ok: bool=True):
        raise NotImplementedError

    @abstractmethod
    def remove_dir(self, dirname: str):
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, filename: str):
        raise NotImplementedError

class FilesysCheckpointStorage(BaseCheckpointStorage):

    def __init__(self, dirname: str):
        super().__init__(dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def dir_exists(self, dirname: str):
        dirname = os.path.join(self._dirname, dirname)
        return os.path.exists(dirname) and os.path.isdir(dirname)

    def file_exists(self, filename: str):
        filename = os.path.join(self._dirname, filename)
        return os.path.exists(filename) and os.path.isfile(filename)

    def is_checkpoint_xser(self, ckpt_path: str):
        ckpt_path = os.path.join(self._dirname, ckpt_path)
        for x in os.listdir(ckpt_path):
            inner_path = os.path.join(ckpt_path, x)
            if os.path.isdir(inner_path):
                for y in os.listdir(inner_path):
                    if y.endswith(".tensors"):
                        return True
        return False

    def list_checkpoint_tags(self):
        '''
        list all the checkpoints under a direcotry
        return a list of check point tags ordered by time of creation
        '''
        if not os.path.exists(self._dirname):
            return []

        dir_entries = sorted(
            (entry for entry in os.scandir(self._dirname) if entry.is_dir() and os.path.exists(os.path.join(entry.path, "done"))),
            key=os.path.getctime
        )
        return [entry.name for entry in dir_entries]

    def save_text(self, text: str, filename: str):
        filename = os.path.join(self._dirname, filename)
        with open(filename, "w") as f:
            f.write(text)

    def save_object(self, obj: object, filename: str):
        filename = os.path.join(self._dirname, filename)
        torch.save(obj, filename)

    def load_object(self, filename: str, map_location=None):
        filename = os.path.join(self._dirname, filename)
        return torch.load(filename, map_location=map_location)

    def remove_dir(self, dirname: str):
        dirname = os.path.join(self._dirname, dirname)
        shutil.rmtree(dirname)

    def create_dir(self, dirname: str, exist_ok: bool=True):
        dirname = os.path.join(self._dirname, dirname)
        os.makedirs(dirname, exist_ok=exist_ok)

    def remove_file(self, filename: str):
        filename = os.path.join(self._dirname, filename)
        os.unlink(filename)


class S3CheckpointStorage(BaseCheckpointStorage):

    def __init__(self, dirname: str):
        super().__init__(dirname)
        self._bucket, self._base_key = S3CheckpointStorage.parse_path(dirname)
        if self._base_key and not self._base_key.endswith("/"):
            self._base_key += "/"

    def dir_exists(dirname: str):
        key = self.convert_path_to_key(dirname)

        client = S3CheckpointStorage.get_client()
        key = key.rstrip('/') 
        response = client.list_objects(Bucket=self._bucket, Prefix=key, Delimiter="/", MaxKeys=1)
        return 'CommonPrefixes' in response

    def file_exists(self, filename: str):
        key = self.convert_path_to_key(filename)

        client = S3CheckpointStorage.get_client()
        try:
            response = client.head_object(Bucket=self._bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def is_checkpoint_xser(self, dirname: str):
        key = self.convert_path_to_key(dirname)
        if not key.endswith("/"):
            key += "/"

        s3 = S3CheckpointStorage.get_client()
        objects = s3.list_objects(Bucket=self._bucket, Prefix=key, Delimiter="/")["CommonPrefixes"]

        for obj in objects:
            inner_key = obj["Prefix"]
            response = s3.list_objects(Bucket=self._bucket, Prefix=inner_key, Delimiter="/")
            if not "CommonPrefixes" in response:
                continue
            inner_objects = response["CommonPrefixes"]
            for inner_object in inner_objects:
                inner_prefix = inner_object["Prefix"]
                if inner_prefix.endswith(".tensors/"):
                    return True
        return False

    def list_checkpoint_tags(self):
        s3 = S3CheckpointStorage.get_client()
        if self._base_key is not None:
            objects = s3.list_objects(Bucket=self._bucket, Prefix=self._base_key, Delimiter="/")["CommonPrefixes"]
        else:
            objects = s3.list_objects(Bucket=self._bucket, Delimiter="/")["CommonPrefixes"]

        tag_date_pairs = []
        for obj in objects:
            inner_dir = obj["Prefix"]
            assert inner_dir[-1] == '/'
            try:
                response = s3.head_object(Bucket=self._bucket, Key=inner_dir + "done")
                base_len = len(self._base_key) if self._base_key is not None else 0
                tag = inner_dir[base_len:-1]     # use -1 to remove the trailing "/"
                date = response["LastModified"]
                tag_date_pairs.append((tag, date))
            except botocore.exceptions.ClientError:
                pass

        tag_date_pairs.sort(key=lambda pair: pair[1])
        tags = [ pair[0] for pair  in tag_date_pairs ]
        return tags

    def save_text(self, text: str, filename: str):
        stream = BytesIO()
        stream.write(bytes(text, "utf-8"))
        self.upload_stream_to_file(stream, filename)

    def save_object(self, obj: object, filename: str):
       stream = BytesIO()
       torch.save(obj, stream)
       self.upload_stream_to_file(stream, filename)

    def load_object(self, filename, map_location=None):
        stream: BytesIO = self.download_file_to_stream(filename)
        return torch.load(stream, map_location=map_location)

    def create_dir(self, dirname: str, exist_ok: bool=True):
        '''
        s3 allow create files with common prefix at the same time, therefore
        nothing need to be done here.
        '''
        pass

    def remove_dir(self, dirname: str):
        key = self.convert_path_to_key(dirname)

        s3 = S3CheckpointStorage.get_resource()
        s3_bucket = s3.Bucket(self._bucket)
        s3_bucket.objects.filter(Prefix=key + "/").delete()

    def upload_stream_to_file(self, stream: BytesIO, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 10) -> None:
        client = S3CheckpointStorage.get_client()
        chunk_size = chunk_size_MB * 1048576
        config = boto3.s3.transfer.TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        stream.seek(0)
        key = self.convert_path_to_key(filename)
        client.upload_fileobj(stream, self._bucket, key, Config=config)

    def remove_file(self, filename: str):
        key = self.convert_path_to_key(filename)

        s3 = S3CheckpointStorage.get_resource()
        s3_object = s3.Object(self._bucket, key)
        s3_object.delete()

    def convert_path_to_key(self, path: str):
        return path if self._base_key is None else self._base_key + path

    def download_file_to_stream(self, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 15) -> BytesIO:
        stream = BytesIO()
        client = S3CheckpointStorage.get_client()
        key = self.convert_path_to_key(filename)
        chunk_size = chunk_size_MB * 1048576
        config = boto3.s3.transfer.TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        client.download_fileobj(self._bucket, key, stream, Config=config)
        stream.seek(0)
        return stream

    @staticmethod
    def parse_path(s3_path: str) -> Tuple[str, str]:
        head = "s3://"
        if not s3_path.startswith(head):
            raise RuntimeError(f"Error: invalid s3 path: {s3_path} because it does not start with {head}")

        s3_path = s3_path[len(head):]
        if len(s3_path) == 0:
            raise RuntimeError("Error: invalid s3 path: {s3_path} that is empty")

        first_slash = s3_path.find("/")
        if first_slash == -1:
            return s3_path, None

        if first_slash == len(s3_path) - 1:
           return s3_path[0:-1], None

        return s3_path[0 : first_slash], s3_path[first_slash + 1:]

    @staticmethod
    def get_resource(profile: str = None, creds: botocore.credentials.Credentials = None, session=None, config={}):
        config=botocore.config.Config(max_pool_connections=30, **config)

        if profile is not None and creds is not None:
            raise ValueError('Please provide profile or creds or neither, not both.')

        if profile is not None:
            s3 = boto3.Session(profile_name=profile).resource('s3', config=config)
        elif creds is not None:
            s3 = boto3.Session().resource('s3',
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                config=config,
            )
        else:
            s3 = boto3.Session().resource('s3', config=config) if not session else session.resource('s3', config=config)

        return s3

    @staticmethod
    def get_client(profile: str = None, creds: botocore.credentials.Credentials = None, session=None, config={}):
        return S3CheckpointStorage.get_resource(profile, creds, session, config).meta.client


def create_checkpoint_storage(dirname: str):
    return S3CheckpointStorage(dirname) if dirname.startswith("s3://") else FilesysCheckpointStorage(dirname)
