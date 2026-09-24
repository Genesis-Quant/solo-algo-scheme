from collections.abc import Iterator
from contextlib import contextmanager

import dolphindb as ddb

from scheme.config import DolphinSettings


@contextmanager
def create_session(settings: DolphinSettings) -> Iterator[ddb.session]:
    session = ddb.session()
    try:
        if not session.connect(
            settings.host, settings.port, settings.username, settings.password.get_secret_value()
        ):
            raise ConnectionError("DolphinDB connection failed")
        yield session
    finally:
        session.close()
