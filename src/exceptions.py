class EasydataError(Exception):
    """General Easydata Error. Further error types are subclassed from this Exception"""
    pass

class ParameterError(EasydataError):
    """Parameter(s) to a function or method are invalid

    This can be used instead of a ValueError or TypeError"""
    pass

class ValidationError(EasydataError):
    """Hash check failed"""
    pass

class ObjectCollision(EasydataError):
    """Object already exists in object store

    This is more general than a FileExistsError, as it applies to more than just the filesystem.
    """
    pass

class NotFoundError(EasydataError):
    """Named object not found in object store

    This is more general than a FileNotFoundError, as it applies to more than just the filesystem.
    """
    pass
