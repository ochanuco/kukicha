class InvalidParams(Exception):
    """引数が不正。JSON-RPC の -32602 で返す。"""


class ToolError(Exception):
    """tools/call の結果を isError: true にして、メッセージを呼び出し元に返す。"""


class EngineBusy(ToolError):
    def __init__(self) -> None:
        super().__init__("busy")
