from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import urllib.parse

# 从 .env 里加载环境变量（如果有）
load_dotenv()


@dataclass
class Config:
    # ====== 通用路径配置 ======
    # 原始 CSV / 其它数据文件所在目录
    data_dir: str = os.getenv("DATA_DIR", "./data")
    # 输出结果（CSV、日志等）目录
    out_dir: str = os.getenv("OUT_DIR", "./out")

    # 兼容旧代码：以前结果写到本地 sqlite 文件
    sqlite_path: str = os.getenv("SQLITE_PATH", "./out/data.sqlite")

    # CSV 读写参数
    csv_sep: str = os.getenv("CSV_SEP", ",")
    csv_encoding: str = os.getenv("CSV_ENCODING", "utf-8")

    # ====== SQL Server (MSSQL) 配置 ======
    # 服务器地址 / 实例名，例如 "localhost\\SQLEXPRESS" 或 "YOUR_SERVER\\INSTANCE"
    mssql_server: str = os.getenv("MSSQL_SERVER", "localhost\\SQLEXPRESS")
    # 数据库名称，例如 "PCMI"
    mssql_database: str = os.getenv("MSSQL_DATABASE", "PCMI")

    # 登录账号（如果用 Windows 身份验证，可以留空，后面自己在连接函数里处理）
    mssql_user: str = os.getenv("MSSQL_USER", "sa")
    mssql_password: str = os.getenv("MSSQL_PASSWORD", "YourStrong!Passw0rd")

    # ODBC 驱动名称，注意和你本机安装的驱动一致
    mssql_driver: str = os.getenv(
        "MSSQL_DRIVER",
        "ODBC Driver 17 for SQL Server"
    )

    # ====== 方便 pyodbc 使用的连接串 ======
    @property
    def mssql_odbc_conn_str(self) -> str:
        """
        给 pyodbc.connect(...) 用的 ODBC 连接字符串。
        用法示例：
            import pyodbc
            from config import config
            conn = pyodbc.connect(config.mssql_odbc_conn_str)
        """
        return (
            f"DRIVER={{{self.mssql_driver}}};"
            f"SERVER={self.mssql_server};"
            f"DATABASE={self.mssql_database};"
            f"UID={self.mssql_user};PWD={self.mssql_password};"
        )

    # ====== 方便 SQLAlchemy / pandas.to_sql 使用的 URL ======
    @property
    def mssql_sqlalchemy_url(self) -> str:
        """
        给 SQLAlchemy create_engine(...) 用的 URL。
        用法示例：
            from sqlalchemy import create_engine
            from config import config
            engine = create_engine(config.mssql_sqlalchemy_url)
        """
        driver_enc = urllib.parse.quote_plus(self.mssql_driver)
        pwd_enc = urllib.parse.quote_plus(self.mssql_password)
        return (
            f"mssql+pyodbc://{self.mssql_user}:{pwd_enc}"
            f"@{self.mssql_server}/{self.mssql_database}"
            f"?driver={driver_enc}"
        )


# 全局单例，方便直接 from config import config 使用
config = Config()
