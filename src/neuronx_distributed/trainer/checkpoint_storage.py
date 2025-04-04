import fnmatch
import glob
import logging
import os
import random
import shutil
import time
import requests
from abc import abstractmethod
from boto3.s3.transfer import TransferConfig
from botocore.session import Session
from io import BytesIO
from tempfile import NamedTemporaryFile
from tenacity import (before_sleep_log, retry, retry_if_exception,
                      stop_after_attempt, RetryCallState)
from tenacity.wait import wait_base
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING

import boto3
import botocore
import torch
import torch_xla.core.xla_model as xm
import math

try:
    import s3transfer
    import s3transfer.crt
    import awscrt

    use_crt = True
except ImportError:
    use_crt = False

from neuronx_distributed.utils.logger import get_logger
logger = get_logger()

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3ServiceResource, S3Client
    from mypy_boto3_s3.type_defs import ListObjectsV2OutputTypeDef

# bytes in MB
MB = 1024 ** 2
# glibc linux system expects POSIX shared memory to be mounted at /dev/shm
SHM_PATH = '/dev/shm'

class BaseCheckpointStorage:
    def __init__(self, dirname: str):
        self._dirname = dirname

    def dirname(self) -> str:
        return self._dirname

    def is_checkpoint_xser(self, dirname: str) -> bool:
        dirs = self.find_subdirs_contain_path(
            pattern="*.tensors", search_depth=2, search_root=dirname, max_count=1, sort_by_mdate=False
        )
        return len(dirs) > 0

    def list_checkpoint_tags(self) -> List[str]:
        return self.find_subdirs_contain_path(pattern="checkpoint", search_depth=1, sort_by_mdate=True)

    def list_completed_checkpoint_tags(self) -> List[str]:
        return self.find_subdirs_contain_path(pattern="done", search_depth=1, sort_by_mdate=True)

    def find_subdirs_contain_path(
        self,
        pattern: str,
        search_depth: int,
        search_root: Optional[str] = None,
        max_count: Optional[int] = None,
        sort_by_mdate: bool = False,
    ) -> List[str]:
        files = self.find_files(pattern, search_depth + 1, search_root, max_count, sort_by_mdate)
        subdirs = []
        for file in files:
            subdirs.append(os.path.dirname(file))
        return subdirs

    @abstractmethod
    def dir_exists(self, dirname: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def file_exists(self, filename: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def find_files(
        self,
        pattern: str,
        search_depth: int,
        search_root: Optional[str] = None,
        max_count: Optional[int] = None,
        sort_by_mdate: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def save_text(self, text: str, filename: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_object(self, obj: object, filename: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_object(self, filename: str, map_location: torch.serialization.MAP_LOCATION = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_dir(self, dirname: str, exist_ok: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_dir(self, dirname: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, filename: str) -> None:
        raise NotImplementedError

    def remove_dirs(self, dirnames: List[str]) -> None:
        for dirname in dirnames:
            self.remove_dir(dirname)

    def remove_files(self, filenames: List[str]) -> None:
        for filename in filenames:
            if self.file_exists(filename):
                self.remove_file(filename)


class FilesysCheckpointStorage(BaseCheckpointStorage):
    def __init__(self, dirname: str):
        super().__init__(dirname)

    def dir_exists(self, dirname: str) -> bool:
        dirname = os.path.join(self._dirname, dirname)
        return os.path.exists(dirname) and os.path.isdir(dirname)

    def file_exists(self, filename: str) -> bool:
        filename = os.path.join(self._dirname, filename)
        return os.path.exists(filename) and os.path.isfile(filename)

    def is_checkpoint_xser(self, ckpt_path: str) -> bool:
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
        search_root: Optional[str] = None,
        max_count: Optional[int] = None,
        sort_by_mdate: bool = False,
    ) -> List[str]:
        if not os.path.exists(self._dirname):
            return []

        if search_root:
            pattern = f"{self._dirname}/{search_root}/**/{pattern}"
        else:
            pattern = f"{self._dirname}/**/{pattern}"

        paths = glob.glob(pattern)
        if sort_by_mdate:
            paths.sort(key=os.path.getmtime)

        if isinstance(max_count, int) and max_count > 0 and len(paths) > max_count:
            paths = paths[0:max_count]

        files = []
        for path in paths:
            files.append(os.path.relpath(path, self._dirname))
        return files

    def save_text(self, text: str, filename: str) -> None:
        filename = os.path.join(self._dirname, filename)
        with open(filename, "w") as f:
            f.write(text)

    def save_object(self, obj: Any, filename: str) -> None:
        filename = os.path.join(self._dirname, filename)
        torch.save(obj, filename)

    def load_object(self, filename: str, map_location: torch.serialization.MAP_LOCATION = None) -> Any:
        filename = os.path.join(self._dirname, filename)
        return torch.load(filename, map_location=map_location, weights_only=False)

    def remove_dir(self, dirname: str) -> None:
        dirname = os.path.join(self._dirname, dirname)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    def remove_file(self, filename: str) -> None:
        filename = os.path.join(self._dirname, filename)
        if os.path.exists(filename):
            os.unlink(filename)

    def create_dir(self, dirname: str, exist_ok: bool = True) -> None:
        dirname = os.path.join(self._dirname, dirname)
        os.makedirs(dirname, exist_ok=exist_ok)

    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> None:
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

class wait_decrementing_with_jitter(wait_base):
    def __init__(self, max_sleep: Union[int, float]) -> None:
        self.max_sleep = max_sleep

    def __call__(self, retry_state: RetryCallState) -> float:
        return random.randint(1, math.ceil(self.max_sleep/retry_state.attempt_number))

# Keeping these variables as globals as they can't be pickled and cause issues
# when checkpoint-saving tries to pickle S3CheckpointStorage as part of multiprocess
# parallelization.
_s3_resource = None
_s3_client = None
_s3_transfer_manager = None

def is_slow_down_error(exception):
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
    if ("<Code>SlowDown</Code>" in message or
        "<Code>RequestTimeout</Code>" in message or
        "<Code>InternalError</Code>" in message):
        return True

    if use_crt and isinstance(exception, awscrt.exceptions.AwsCrtError):
        return True

    if isinstance(exception, botocore.exceptions.ConnectionClosedError):
        if "Connection was closed before we received a valid response from endpoint" in message and ".s3." in message:
            return True

    if isinstance(exception, botocore.exceptions.ClientError):
        if exception.response:
            if 'Error' not in exception.response:
                message = str(exception.response)
                return "MaxAttemptsReached" in message
            else:
                error_code = exception.response['Error']['Code']
                return error_code in ['SlowDown', 'RequestTimeout', 'InternalError', 'Throttling']

    return False

class S3CheckpointStorage(BaseCheckpointStorage):
    S3_PATH_PREFIX = 's3://'

    retry_with_jitter = retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception(is_slow_down_error),
        wait=wait_decrementing_with_jitter(max_sleep=int(os.environ.get("WORLD_SIZE", "50000")) / 10000),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )

    def __init__(self, dirname: str, crt_config={}):
        super().__init__(dirname)
        self._bucket, self._base_key = S3CheckpointStorage.parse_path(dirname)
        if self._base_key and not self._base_key.endswith("/"):
            self._base_key += "/"

        # TODO @jaczhao: Support user config to pass crt configs here. Only cherry picking checkpoint_storage changes for now
        self._s3_crt_upload_part_size_mb = crt_config.get("s3_crt_upload_part_size_mb")
        self._s3_crt_upload_num_threads = crt_config.get("s3_crt_upload_num_threads")

        boto3.set_stream_logger(name="botocore.credentials", level=logging.ERROR)

    def dir_exists(self, dirname: str) -> bool:
        """
        s3 allow create files with common prefix at the same time, therefore
        we can consider any diretory to be existing
        """
        return True

    @retry_with_jitter
    def file_exists(self, filename: str) -> bool:
        try:
            S3CheckpointStorage.get_client().head_object(Bucket=self._bucket, Key=self.convert_path_to_key(filename))
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            raise

    @retry_with_jitter
    def _list(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        s3 = S3CheckpointStorage.get_client()

        list_prefix = os.path.join(self._base_key, prefix) if self._base_key and prefix else (self._base_key or prefix)

        response: "ListObjectsV2OutputTypeDef"
        if list_prefix:
            if list_prefix[-1] != "/":
                list_prefix += "/"  # list_object_v2 require prefix to be end with '/'
            response = s3.list_objects_v2(Bucket=self._bucket, Prefix=list_prefix, Delimiter="/")
        else:
            response = s3.list_objects_v2(Bucket=self._bucket, Delimiter="/")

        results = []
        prefix_len = len(list_prefix) if list_prefix else 0
        if "Contents" in response:
            for item in response["Contents"]:
                results.append({"type": "file", "name": item["Key"][prefix_len:], "mdate": item["LastModified"]})

        if "CommonPrefixes" in response:
            for common_prefix in response["CommonPrefixes"]:
                results.append({"type": "dir", "name": common_prefix["Prefix"][prefix_len:-1]})

        return results


    def _find_files_impl(self, pattern: str, search_depth: int, search_root: Optional[str], max_count: Optional[int]):
        search_dirs = [search_root]
        search_bgn = 0
        search_end = 1
        file_mdate_pairs = []
        level = 0
        while level <= search_depth and search_bgn < search_end:
            for i in range(search_bgn, search_end):
                dirname = search_dirs[i]
                paths = self._list(dirname)
                for path in paths:
                    if fnmatch.fnmatch(path["name"], pattern):
                        mdate = path.get("mdate", None)
                        assert dirname
                        file_mdate_pairs.append((os.path.join(dirname, path["name"]), mdate))
                        if max_count and max_count > 0 and len(file_mdate_pairs) == max_count:
                            return file_mdate_pairs
                    elif path["type"] == "dir":
                        subdir = os.path.join(dirname, path["name"]) if dirname else path["name"]
                        search_dirs.append(subdir)

            search_bgn = search_end
            search_end = len(search_dirs)
            level += 1
        return file_mdate_pairs

    def find_files(
        self,
        pattern: str,
        search_depth: int,
        search_root: Optional[str] = None,
        max_count: Optional[int] = None,
        sort_by_mdate: bool = True,
    ) -> List[str]:
        file_mdate_pairs = self._find_files_impl(pattern, search_depth, search_root, max_count)
        if len(file_mdate_pairs) > 1 and sort_by_mdate:
            file_mdate_pairs.sort(key=lambda x: x[1])

        files = [x[0] for x in file_mdate_pairs]
        return files

    def save_text(self, text: str, filename: str, use_threads: bool = True) -> None:
        class TextStreamCreator:
            def __init__(self, text: str):
                self._text = text

            def create_stream(self) -> BytesIO:
                stream = BytesIO()
                stream.write(bytes(self._text, "utf-8"))
                return stream

            def create_tmp_file(self):
                try:
                    tempfile = NamedTemporaryFile(dir=SHM_PATH, mode='w', delete=False)
                    tempfile.write(text)
                    return tempfile.name
                finally:
                    tempfile.close()

        self.upload_stream_to_file(TextStreamCreator(text), filename, 64, 10, use_threads)

    def save_object(self, obj: object, filename: str) -> None:
        class ObjectStreamCreator:
            def __init__(self, obj: object):
                self._obj = obj

            def create_stream(self) -> BytesIO:
                stream = BytesIO()
                torch.save(obj, stream)
                return stream

            def create_tmp_file(self):
                try:
                    tempfile = NamedTemporaryFile(dir=SHM_PATH, delete=False)
                    torch.save(self._obj, tempfile)
                    return tempfile.name
                finally:
                    tempfile.close()

        self.upload_stream_to_file(ObjectStreamCreator(obj), filename)

    def load_object(self, filename: str, map_location: Optional[torch.serialization.MAP_LOCATION] = None) -> Any:
        stream: BytesIO = self.download_file_to_stream(filename)
        return torch.load(stream, map_location=map_location, weights_only=False)

    def create_dir(self, dirname: str, exist_ok: bool = True) -> None:
        """
        s3 allow create files with common prefix at the same time, therefore
        nothing need to be done here.
        """

    def create_shared_dir(
        self, dirname: str, exist_ok: bool = True, process_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> None:
        """
        s3 allow create files with common prefix at the same time, therefore
        nothing need to be done here.
        """

    @retry_with_jitter
    def remove_dir(self, dirname: str) -> None:
        key = self.convert_path_to_key(dirname)
        s3 = S3CheckpointStorage.get_resource()
        s3_bucket = s3.Bucket(self._bucket)
        s3_bucket.objects.filter(Prefix=key + "/").delete()

    @retry_with_jitter
    def remove_file(self, filename: str) -> None:
        key = self.convert_path_to_key(filename)
        s3 = S3CheckpointStorage.get_resource()
        s3_object = s3.Object(self._bucket, key)
        s3_object.delete()

    @retry_with_jitter
    def upload_stream_to_file(
            self, stream_creator, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 10, use_threads: bool = True
    ) -> None:
        chunk_size = chunk_size_MB * 1048576
        ckpt_fileobj = None
        tmpfile_buffer = False
        if os.path.exists(SHM_PATH) and use_threads:
            tmpfile_buffer = True
            ckpt_fileobj = stream_creator.create_tmp_file()
        else:
            ckpt_fileobj = stream_creator.create_stream()
            ckpt_fileobj.seek(0)
        config = TransferConfig(use_threads=use_threads, multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        key = self.convert_path_to_key(filename)
        if use_threads:
            manager = self.get_transfer_manager(config)
            future = manager.upload(
                fileobj=ckpt_fileobj,
                bucket=self._bucket,
                key=key,
            )
            future.result()
            if tmpfile_buffer:
                os.remove(ckpt_fileobj)
        else: 
            # if use_threads is False, then we need to use boto3 client directly as CRT transfer manager doesn't support non-background threading
            client = self.get_client()
            client.upload_fileobj(ckpt_fileobj, self._bucket, key, Config=config)

    def convert_path_to_key(self, path: str) -> str:
        return path if self._base_key is None else self._base_key + path

    @retry_with_jitter
    def download_file_to_stream(self, filename: str, chunk_size_MB: int = 64, max_concurrency: int = 15) -> BytesIO:
        stream = BytesIO()
        # Using get_object to avoid spinning up more threads as download_fileobj does.
        # Downloading the checkpoint tensors happens already from 3 threads per process for 32 processes per node.
        response = S3CheckpointStorage.get_client().get_object(Bucket=self._bucket, Key=self.convert_path_to_key(filename))
        stream.write(response['Body'].read())
        stream.seek(0)
        return stream

    @staticmethod
    def parse_path(s3_path: str) -> Tuple[str, Optional[str]]:
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
    def get_resource() -> "S3ServiceResource":
        s3_resource, _ = S3CheckpointStorage._ensure_s3_resource_and_client()
        return s3_resource

    @staticmethod
    def get_client() -> "S3Client":
        _, s3_client = S3CheckpointStorage._ensure_s3_resource_and_client()
        return s3_client

    @staticmethod
    def _ensure_s3_resource_and_client() -> Tuple["S3ServiceResource", "S3Client"]:
        """Only instantiate the s3 resource and client once per process."""
        global _s3_resource, _s3_client
        if _s3_resource is None or _s3_client is None:
            _s3_resource = boto3.Session().resource('s3', config=botocore.config.Config(max_pool_connections=max(1, (os.cpu_count() or 1) // 4)))
            _s3_client = _s3_resource.meta.client
        return _s3_resource, _s3_client

    @staticmethod
    def _get_s3_region() -> str:
        url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            instance_data = response.json()
            return instance_data['region']
        except requests.exceptions.RequestException as e:
            # Fallback to AWS SDK region environment variable
            logging.info(f"Failed to retrieve instance region: {e}. Defaulting to AWS_REGION value")
            region = os.getenv("AWS_REGION")
            if region is None:
                # If not AWS_REGION not set, fallback to the region that is auto-resolved by boto3 client
                region = S3CheckpointStorage.get_client().meta.region_name
                logging.info(f"AWS_REGION env variable not set. Defaulting to {region}")
            return region


    @staticmethod
    def _create_s3_crt_client(session: botocore.session.Session, num_threads=None, part_size=None, target_throughput=None) -> Any:
        logging.debug(f"CRT client config: part_size={part_size}, num_threads={num_threads}")

        botocore_credentials = session.get_credentials()
        # ignoring type error b/c this is how it is done in CRT integ/unit tests: https://github.com/boto/s3transfer/pull/283/commits/a86f7cf04b2e4dc89a868f5c198b9c057aada7de
        wrapper = s3transfer.crt.BotocoreCRTCredentialsWrapper(
            botocore_credentials  # type: ignore[arg-type] 
        )
        crt_credentials_provider = wrapper.to_crt_credentials_provider()
        s3_crt_client = s3transfer.crt.create_s3_crt_client(
            region=S3CheckpointStorage._get_s3_region(),
            crt_credentials_provider=crt_credentials_provider,
            target_throughput=target_throughput,
            num_threads=num_threads,
            part_size=part_size,
        )
        return s3_crt_client

    def _create_transfer_manager(self, transfer_config: TransferConfig) -> Any:

        crt = use_crt and awscrt.__version__ > "0.19.18" # type: ignore[attr-defined]
        if crt: 
            config = {}
            if self._s3_crt_upload_num_threads:
                config['num_threads'] = int(self._s3_crt_upload_num_threads)
            if self._s3_crt_upload_part_size_mb:
                config['part_size'] = int(float(self._s3_crt_upload_part_size_mb) * MB)
            session = Session()
            s3_crt_client = S3CheckpointStorage._create_s3_crt_client(session, **config)
            request_serializer = s3transfer.crt.BotocoreCRTRequestSerializer(session)
            crt_transfer_manager = s3transfer.crt.CRTTransferManager(s3_crt_client, request_serializer)
            return crt_transfer_manager
        s3_client = S3CheckpointStorage.get_client()
        return s3transfer.manager.TransferManager(s3_client, transfer_config)

    def get_transfer_manager(self, config: Optional[TransferConfig]=None) -> Any:
        global _s3_transfer_manager
        if _s3_transfer_manager is None:
            if config is None:
                config = TransferConfig()
            _s3_transfer_manager = self._create_transfer_manager(config)
        return _s3_transfer_manager


def create_checkpoint_storage(dirname: str, crt_config: Optional[Dict]={}):
    return S3CheckpointStorage(dirname, crt_config) if dirname.startswith(S3CheckpointStorage.S3_PATH_PREFIX) else FilesysCheckpointStorage(dirname)
