"""
数据加载工具 — 读取 MD 文件和 Excel/CSV 数据
"""

import pandas as pd


def read_md_file(file_path: str) -> str:
    """读取MD文件，返回文本内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # print(f"✅ 读取成功: {file_path} ({len(content)} 字符)")
        return content
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return ""
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")
        return ""


def load_hongkong_data(file_path: str = "hongkong_llm_data.xlsx") -> pd.DataFrame:
    """加载香港本地数据（支持Excel和CSV，支持多种编码）"""

    # 判断文件类型
    file_extension = file_path.split('.')[-1].lower()

    # Excel文件
    if file_extension in ['xlsx', 'xls']:
        try:
            print(f"🔄 尝试读取Excel文件: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl' if file_extension == 'xlsx' else None)
            print(f"✅ 成功读取Excel文件!")
            print(f"📊 数据形状: {df.shape}")
            print(f"📋 列名: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
            return df
        except Exception as e:
            print(f"⚠️ 读取Excel文件失败: {e}")
            # 尝试其他sheet
            try:
                print("🔄 尝试读取第一个sheet...")
                df = pd.read_excel(file_path, sheet_name=0)
                print(f"✅ 成功读取第一个sheet!")
                print(f"📊 数据形状: {df.shape}")
                return df
            except Exception as e2:
                print(f"⚠️ 读取第一个sheet也失败: {e2}")

    # CSV文件
    elif file_extension == 'csv':
        # 尝试多种编码方式
        encodings = ['utf-8', 'big5', 'big5hkscs', 'gb18030', 'gbk', 'gb2312', 'utf-16', 'latin1']

        for encoding in encodings:
            try:
                print(f"🔄 尝试使用 {encoding} 编码读取CSV...")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ 成功使用 {encoding} 编码读取文件!")
                print(f"📊 数据形状: {df.shape}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:
                    print(f"⚠️ 所有编码尝试失败: {e}")

        # 尝试自动检测编码
        try:
            print("🔄 尝试自动检测编码...")
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding']
                confidence = result['confidence']
                print(f"🔍 检测到编码: {detected_encoding} (置信度: {confidence:.2%})")

            df = pd.read_csv(file_path, encoding=detected_encoding)
            print(f"✅ 使用检测的编码成功读取!")
            return df
        except ImportError:
            print("⚠️ 需要安装 chardet: pip install chardet")
        except Exception as e:
            print(f"⚠️ 自动检测也失败: {e}")

    else:
        print(f"⚠️ 不支持的文件格式: {file_extension}")

    # 返回示例数据
    print("⚠️ 返回示例数据用于测试")
    return pd.DataFrame({
        "選區(中文名稱)": ["油尖旺", "中西區", "灣仔"],
        "平均收入": [25000, 45000, 38000],
        "人口密度": [45000, 52000, 41000],
        "平均年齡": [42, 38, 40]
    })
