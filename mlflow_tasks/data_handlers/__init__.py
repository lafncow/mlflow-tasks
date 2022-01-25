from .py_obj_handler import Py_Obj_Handler
from .pandas_df_handler import Pandas_Df_Handler

data_handlers = {
    "default": Py_Obj_Handler,
    "py_obj": Py_Obj_Handler,
    "pandas_df": Pandas_Df_Handler
}