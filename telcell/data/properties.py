class Properties(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return Properties()

    def __repr__(self):
        return super().__repr__()


class LocationInfo(Properties):
    pass
