"""Read optional immutable upload metadata from SFTP."""

import errno
import json


def remote_json(sftp, path):
    try:
        with sftp.open(path, "rb") as handle:
            return json.loads(handle.read())
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise
