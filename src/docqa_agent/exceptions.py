class DocQaError(Exception):
    pass


class ConfigurationError(DocQaError):
    pass


class IndexNotFoundError(DocQaError):
    pass


class ParsingError(DocQaError):
    pass
