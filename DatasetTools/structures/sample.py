from typing import List, Optional

from DatasetTools.structures.image import Image
from DatasetTools.structures.instance import Instance


class Sample:

    def __init__(
        self,
        image: Optional[Image] = None,
        annotations: Optional[List[Instance]] = None
    ):
        self.image = image
        self.annotations = annotations
