import fnmatch
import glob
import logging
import os
import random
import shutil
import time
from abc import abstractmethod
from io import BytesIO
from typing import List, Tuple

import boto3
import botocore
import torch
import torch_xla.core.xla_model as xm

try:
    import awscrt

    use_crt = True
except ImportError:
    use_crt = False


class BaseCheckpointStorage:
    def __init__(self, dirname: str):
        self._dirname = dirname

    def dirname(self):
        return self._dirname

    def is_checkpoint_xser(self, dirname: str):
        dirs = self.find_subdirs_contain_path(
            pattern="*.tensors", search_depth=2, search_root=dirname, max_count=1, sort_by_mdate=False
        )
        return len(dirs) > 0

    def list_checkpoint_tags(self):
        return self.find_subdirs_contain_path(pattern="checkpoint", search_depth=1, sort_by_mdate=True)

    def list_completed_checkpoint_tags(self):
        return self.find_subdirs_contain_path(pattern="done", search_depth=1, sort_by_mdate=True)

    def find_subdirs_contain_path(
        self,
        pattern: str,
        search_depth: int,
        search_root: str = None,
        max_count: int = None,
        sort_by_mdate: bool = False,
    ):
        files = self.find_files(pattern, search_depth + 1, search_root, max_count, sort_by_mdate)
        subdirs = []
        for file in files:
            subdirs.append(os.path.dirname(file))
        return subdirs

    @abstractmethod
    def dir_exists(self, dirname: str):
        raise NotImplementedError

    @abstractmethod
    def file_exists(self, filename: str):
        raise NotImplementedError

    @abstractmethod
    def find_files(
        self,
        pattern: str,
        search_depth: int,
        search_root: str = None,
        max_count: int = None,
        sort_by_mdate: bool = False,
    ):
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
    def create_dir(self, dirname: str, exist_ok: bool = True):
        raise NotImplementedError

    @abstractmethod
    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: torch.distributed.ProcessGroup = None
    ):
        raise NotImplementedError

    @abstractmethod
    def remove_dir(self, dirname: str):
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, filename: str):
        raise NotImplementedError

    def remove_dirs(self, dirnames: List[str]):
        for dirname in dirnames:
            self.remove_dir(dirname)

    def remove_files(self, filenames: List[str]):
        for filename in filenames:
            if self.file_exists(filename):
                self.remove_file(filename)


class FilesysCheckpointStorage(BaseCheckpointStorage):
    def __init__(self, dirname: str):
        super().__init__(dirname)

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

    def find_files(
        self,
        pattern: str,
        search_depth: int,
        search_root: str = None,
        max_count: int = None,
        sort_by_mdate: bool = False,
    ):
        if not os.path.exists(self._dirname):
            return []

        if search_root:
            pattern = f"{self._dirname}/{search_root}/**/{pattern}"
        else:
            pattern = f"{self._dirname}/**/{pattern}"

        paths = glob.glob(pattern)
        if sort_by_mdate:
            paths.sort(key=os.path.getmtime)

        if type(max_count) == int and max_count > 0 and len(paths) > max_count:
            paths = paths[0:max_count]

        files = []
        for path in paths:
            files.append(os.path.relpath(path, self._dirname))
        return files

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
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    def remove_file(self, filename: str):
        filename = os.path.join(self._dirname, filename)
        if os.path.exists(filename):
            os.unlink(filename)

    def create_dir(self, dirname: str, exist_ok: bool = True):
        dirname = os.path.join(self._dirname, dirname)
        os.makedirs(dirname, exist_ok=exist_ok)

    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: torch.distributed.ProcessGroup = None
    ):
        if process_group is None:
            return self.create_dir(dirname, exist_ok)

        if exist_ok:
            # let all processes in the group to create directory,
            # the first one wins. The losers may throw an exception
            # about directory already exist, which we ignore.
            try:
                self.create_dir(dirname, exist_ok=True)
            except OSError:
                pass
        else:
            if process_group.rank() == 0:
                self.create_dir(dirname, exist_ok)
            # TODO: use torch.distributed_barrier(group=process_group)
            # once xla backend support barrier
            xm.rendezvous("create shared dir")


class S3CheckpointStorage(BaseCheckpointStorage):
    UPLOAD = 1
    DOWNLOAD = 2
    REMOVE_DIR = 3
    REMOVE_FILE = 4

    def __init__(self, dirname: str):
        super().__init__(dirname)
        self._bucket, self._base_key = S3CheckpointStorage.parse_path(dirname)
        if self._base_key and not self._base_key.endswith("/"):
            self._base_key += "/"

        boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)

    def dir_exists(self, dirname: str):
        """
        s3 allow create files with common prefix at the same time, therefore
        we can consider any diretory to be existing
        """
        return True

    def file_exists(self, filename: str):
        subdir = os.path.dirname(filename)
        basename = os.path.basename(filename)
        if subdir == "":
            subdir = None

        paths = self._list_with_retry(subdir)
        for path in paths:
            if path["name"] == basename and path["type"] == "file":
                return True

        return False

    def _list(self, prefix: str = None):
        s3 = S3CheckpointStorage.get_client()

        if self._base_key and prefix:
            list_prefix = os.path.join(self._base_key, prefix)
        else:
            list_prefix = self._base_key if self._base_key else prefix

        if list_prefix:
            if list_prefix[-1] != "/":
                list_prefix += "/"  # list_object_v2 require prefix to be end with '/'
            response = s3.list_objects_v2(Bucket=self._bucket, Prefix=list_prefix, Delimiter="/")
        else:
            response = s3.list_objects_v2(Bucket=self._bucket, Delimiter="/")

        results = []
        prefix_len = len(list_prefix) if list_prefix else 0
        if "Contents" in response:
            for obj in response["Contents"]:
                results.append({"type": "file", "name": obj["Key"][prefix_len:], "mdate": obj["LastModified"]})

        if "CommonPrefixes" in response:
            for obj in response["CommonPrefixes"]:
                results.append({"type": "dir", "name": obj["Prefix"][prefix_len:-1]})

        return results

    def _list_with_retry(self, prefix: str = None):
        max_try = 4
        sleep_second = 60
        for try_idx in range(max_try):
            try:
                return self._list(prefix)
            except Exception as e:
                if S3CheckpointStorage.is_slow_down_error(e):
                    if try_idx < max_try - 1:
                        current_sleep_time = sleep_second + random.randint(20, 60)
                        logging.info(
                            f"Encountered slow down error when upload file for {try_idx+1} times. Sleep {current_sleep_time} seconds then retry"
                        )
                        time.sleep(current_sleep_time)
                        sleep_second *= 1.5
                    else:
                        logging.info(
                            f"Encountered slow down error when upload file for {try_idx+1} times. No more retrying"
                        )
                        raise e
                else:
                    raise e

    def _find_files_impl(self, pattern: str, search_depth: int, search_root: str, max_count: int):
        search_dirs = [search_root]
        search_bgn = 0
        search_end = 1
        file_mdate_pairs = []
        level = 0
        while level <= search_depth and search_bgn < search_end:
            for i in range(search_bgn, search_end):
                dirname = search_dirs[i]
                paths = self._list_with_retry(dirname)
                for path in paths:
                    if fnmatch.fnmatch(path["name"], pattern):
                        mdate = path.get("mdate", None)
                        file_mdate_pairs.append((os.path.join(dirname, path["name"]), mdate))
                        if type(max_count) == int and max_count > 0 and len(file_mdate_pairs) == max_count:
                            return file_mdate_pairs
                    elif path["type"] == "dir":
                        subdir = os.path.join(dirname, path["name"]) if dirname else path["name"]
                        search_dirs.append(subdir)

            search_bgn = search_end
            search_end = len(search_dirs)
            level += 1
        return file_mdate_pairs

    def find_files(self, pattern: str, search_depth: int, search_root: str = None, max_count=None, sort_by_mdate=True):
        file_mdate_pairs = self._find_files_impl(pattern, search_depth, search_root, max_count)
        if len(file_mdate_pairs) > 1 and sort_by_mdate:
            file_mdate_pairs.sort(key=lambda x: x[1])

        files = [x[0] for x in file_mdate_pairs]
        return files

    def save_text(self, text: str, filename: str):
        class TextStreamCreator:
            def __init__(self, text):
                self._text = text

            def create_stream(self):
                stream = BytesIO()
                stream.write(bytes(self._text, "utf-8"))
                return stream

        self.upload_stream_to_file(TextStreamCreator(text), filename)

    def save_object(self, obj: object, filename: str):
        class ObjectStreamCreator:
            def __init__(self, obj):
                self._obj = obj

            def create_stream(self):
                stream = BytesIO()
                torch.save(obj, stream)
                return stream

        self.upload_stream_to_file(ObjectStreamCreator(obj), filename)

    def load_object(self, filename, map_location=None):
        stream: BytesIO = self.download_file_to_stream(filename)
        return torch.load(stream, map_location=map_location)

    def create_dir(self, dirname: str, exist_ok: bool = True):
        """
        s3 allow create files with common prefix at the same time, therefore
        nothing need to be done here.
        """

    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: torch.distributed.ProcessGroup = None
    ):
        """
        s3 allow create files with common prefix at the same time, therefore
        nothing need to be done here.
        """

    def remove_dir(self, dirname: str):
        key = self.convert_path_to_key(dirname)
        client = S3CheckpointStorage.get_client()
        S3CheckpointStorage.s3_action_with_retry(
                S3CheckpointStorage.REMOVE_DIR, client, self._bucket, key, None
            )

    def remove_file(self, filename: str):
        key = self.convert_path_to_key(filename)
        client = S3CheckpointStorage.get_client()
        S3CheckpointStorage.s3_action_with_retry(
                S3CheckpointStorage.REMOVE_FILE, client, self._bucket, key, None
            )

    def upload_stream_to_file(
        self, stream_creator, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 10
    ) -> None:
        client = S3CheckpointStorage.get_client()
        chunk_size = chunk_size_MB * 1048576
        config = boto3.s3.transfer.TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        key = self.convert_path_to_key(filename)
        S3CheckpointStorage.s3_action_with_retry(
            S3CheckpointStorage.UPLOAD, client, self._bucket, key, config, upload_stream_creator=stream_creator
        )

    def convert_path_to_key(self, path: str):
        return path if self._base_key is None else self._base_key + path

    def download_file_to_stream(self, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 15) -> BytesIO:
        client = S3CheckpointStorage.get_client()
        key = self.convert_path_to_key(filename)
        chunk_size = chunk_size_MB * 1048576
        config = boto3.s3.transfer.TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        return S3CheckpointStorage.s3_action_with_retry(
            S3CheckpointStorage.DOWNLOAD, client, self._bucket, key, config
        )

    @staticmethod
    def parse_path(s3_path: str) -> Tuple[str, str]:
        head = "s3://"
        if not s3_path.startswith(head):
            raise RuntimeError(f"Error: invalid s3 path: {s3_path} because it does not start with {head}")

        s3_path = s3_path[len(head) :]
        if len(s3_path) == 0:
            raise RuntimeError("Error: invalid s3 path: {s3_path} that is empty")

        first_slash = s3_path.find("/")
        if first_slash == -1:
            return s3_path, None

        if first_slash == len(s3_path) - 1:
            return s3_path[0:-1], None

        return s3_path[0:first_slash], s3_path[first_slash + 1 :]

    @staticmethod
    def get_resource(profile: str = None, creds: botocore.credentials.Credentials = None, session=None, config={}):
        config = botocore.config.Config(max_pool_connections=30, **config)

        if profile is not None and creds is not None:
            raise ValueError("Please provide profile or creds or neither, not both.")

        if profile is not None:
            s3 = boto3.Session(profile_name=profile).resource("s3", config=config)
        elif creds is not None:
            s3 = boto3.Session().resource(
                "s3",
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                config=config,
            )
        else:
            s3 = boto3.Session().resource("s3", config=config) if not session else session.resource("s3", config=config)

        return s3

    @staticmethod
    def get_client(profile: str = None, creds: botocore.credentials.Credentials = None, session=None, config={}):
        return S3CheckpointStorage.get_resource(profile, creds, session, config).meta.client

    @staticmethod
    def is_slow_down_error(exception):
        class_name = exception.__class__.__name__
        module_name = exception.__class__.__module__
        full_class_name = f"{module_name}.{class_name}"

        # Example Invalid response status that is slow down
        # AWS_ERROR_S3_INVALID_RESPONSE_STATUS: Invalid response status from request.
        # Body from error request is:
        # '<?xml version="1.0" encoding="UTF-8"?>
        #  <Error>
        #      <Code>RequestTimeout</Code>
        #      <Message>
        #         Your socket connection to the server was not read from or written to within the timeout period. Idle connections will be closed.
        #      </Message>
        #    <RequestId>XPHS9896G3RJE364</RequestId>
        #    <HostId>ZAiF3HPpUD5IgSr/mfkP2QPs7ttuvY+uTRG9MET/jZZ45MJ6bVbnvSBQLggICvPCROPP/1k85p4=</HostId>
        # </Error>'
        message = str(exception)
        if (
            "<Code>SlowDown</Code>" in message
            or "<Code>RequestTimeout</Code>" in message
            or "<Code>InternalError</Code>" in message
        ):
            return True

        if use_crt and isinstance(exception, awscrt.exceptions.AwsCrtError):
            return True

        if isinstance(exception, botocore.exceptions.ConnectionClosedError):
            if (
                "Connection was closed before we received a valid response from endpoint" in message
                and ".s3." in message
            ):
                return True

        if isinstance(exception, botocore.exceptions.ClientError):
            if exception.response:
                if "Error" not in exception.response:
                    message = str(exception.response)
                    return "MaxAttemptsReached" in message
                else:
                    error_code = exception.response["Error"]["Code"]
                    return error_code in ["SlowDown", "RequestTimeout", "InternalError", "Throttling"]

        return False

    @staticmethod
    def s3_action_with_retry(action, client, bucket, key, config, upload_stream_creator=None):
        max_try = 12
        sleep_second = 60
        for try_idx in range(max_try):
            try:
                if action == S3CheckpointStorage.DOWNLOAD:
                    stream = BytesIO()
                    client.download_fileobj(bucket, key, stream, Config=config)
                    stream.seek(0)
                    return stream
                elif action == S3CheckpointStorage.UPLOAD:
                    stream = upload_stream_creator.create_stream()
                    stream.seek(0)
                    client.upload_fileobj(stream, bucket, key, Config=config)
                    return
                elif action == S3CheckpointStorage.REMOVE_FILE:
                    assert not key.endswith("/")
                    client.delete_object(Bucket=bucket, Key=key)
                    return
                elif action == S3CheckpointStorage.REMOVE_DIR:
                    prefix = key if key.endswith("/") else key + "/"
                    response = client.list_objects(Bucket=bucket, Prefix=prefix)
                    while 'Contents' in response:
                        objects = response['Contents']
                        assert len(objects) > 0
                        delete = {'Objects' : []}
                        for obj in objects:
                             delete['Objects'].append(
                                     {'Key' : obj['Key']}
                                )
                        client.delete_objects(Bucket=bucket, Delete=delete)
                        response = client.list_objects(Bucket=bucket, Prefix=prefix)
                    return
                else:
                    raise RuntimError(f"Error: unknow action {action}")
            except Exception as e:
                if S3CheckpointStorage.is_slow_down_error(e):
                    if try_idx < max_try - 1:
                        current_sleep_time = sleep_second + random.randint(20, 60)
                        logging.info(
                            f"Encountered slow down error when upload file for {try_idx+1} times. Sleep {current_sleep_time} seconds then retry"
                        )
                        time.sleep(current_sleep_time)
                        sleep_second *= 1.5
                    else:
                        logging.info(
                            f"Encountered slow down error when upload file for {try_idx+1} times. No more retrying"
                        )
                        raise e
                else:
                    raise e


def create_checkpoint_storage(dirname: str):
    return S3CheckpointStorage(dirname) if dirname.startswith("s3://") else FilesysCheckpointStorage(dirname)
