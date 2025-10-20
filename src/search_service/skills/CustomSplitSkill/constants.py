from enum import Enum

class FileExtenstions(str,Enum):
    PPTX = ".pptx"
    PPT = ".ppt"
    PDF = ".pdf"
    DOCX = ".docx"
    DOC = ".doc"
    XLSX = ".xlsx"
    XLS = ".xls"
    XLSM = ".xlsm"
    CSV = ".csv"
    PY=".py"
    
    
    

blob_uri_prefix_constant = ".blob.core.windows.net/"

class ColumnNames(str, Enum):
    ITEM_FUNCTION = "Item Function"
    INTENDED_FUNCTION = "Intended Function"
    POTENTIAL_FAILURE_MODE = "Potential Failure Mode"
    POTENTIAL_EFFECTS_OF_FAILURE = "Potential Effect(s) of Failure"
    SEVERITY = "Severity (S)"
    POTENTIAL_CAUSES = "Potential Cause(s) / Mechanism(s) of Failure"
    FAILURE_CATEGORY = "Failure Category"
    CURRENT_DESIGN_PREVENTION_CONTROL = "Current Design Prevention Control & Explanation of Occurrence rating"
    OCCURRENCE = "Occurrence (O)"
    CLASS = "Class"
    CURRENT_DETECTION_DESIGN_CONTROLS = "Current Detection Design Controls & Explanation of detection rating"
    DETECTION = "Detection (D)"
    RPN = "RPN (S*O*D)"
    RECOMMENDED_ACTION = "Recommended Action"
    CORRECTIVE_ACTIONS = "Corrective Actions"
    RECOMMENDED_CORRECTIVE_ACTIONS = "Recommended Corrective Action(s)"