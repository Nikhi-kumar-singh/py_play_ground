import os
import sys
import traceback


class custom_exception(Exception):
    def __init__(
        self,
        error_message,
        error_data: sys
    ):
        super().__init__(error_message)
        self.error_message = error_message
        self.exc_type, exc_obj, tb = error_data.exc_info()
        last_trace = traceback.extract_tb(tb)[-1]
        self.fname = last_trace.filename
        self.line_no = last_trace.lineno

    def __str__(self):
        return (
            f"In file [{self.fname}], "
            f"in line number [{self.line_no}], "
            f"error occurred [{self.error_message}]"
        )


if __name__ == "__main__":
    try:
        a = 10 / 0
    except Exception as e:
        raise custom_exception(e, sys)
