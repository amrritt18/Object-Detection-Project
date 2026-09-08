import sys


def error_message_detail(error, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in Python script "
        f"[{file_name}] at line number [{line_number}] "
        f"with error message [{str(error)}]"
    )


class SignException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(str(error_message))

        self.error_message = error_message_detail(
            error_message,
            error_detail,
        )

    def __str__(self) -> str:
        return self.error_message