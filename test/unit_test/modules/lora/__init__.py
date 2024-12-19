class MockGroup:
    def __init__(self, size=8) -> None:
        self.ranks = [i for i in range(size)]
        self._size = size

    def size(self):
        return self._size
