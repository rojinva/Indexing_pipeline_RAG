import enum

class StrEnum(str, enum.Enum):
    pass


class FileExtensions(StrEnum):
    PPTX = ".pptx"
    PPT = ".ppt"
    PDF = ".pdf"
    DOCX = ".docx"
    DOC = ".doc"
    XLSX = ".xlsx"
    XLS = ".xls"
    XLSM = ".xlsm"
    CSV = ".csv"
