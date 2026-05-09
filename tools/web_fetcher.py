"""
网页内容抓取与 PDF 文本提取工具
"""

import io
import re
import logging
from typing import Tuple

import requests
from bs4 import BeautifulSoup
from langsmith import traceable

logger = logging.getLogger(__name__)


# ==================== PDF 文本提取 ====================

def _is_meaningful_text(text: str, min_chars: int = 50) -> bool:
    """判断提取出的文本是否有意义"""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    printable_pattern = re.compile(
        r'[\u4e00-\u9fff'
        r'\u3400-\u4dbf'
        r'\u0020-\u007e'
        r'\u00a0-\u00ff'
        r'\u2000-\u206f'
        r'\u3000-\u303f'
        r'\uff00-\uffef'
        r'\n\r\t]'
    )
    printable_count = len(printable_pattern.findall(stripped))
    ratio = printable_count / len(stripped)
    return ratio > 0.6


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """pdfplumber: 默认首选"""
    import pdfplumber
    texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
            else:
                logger.debug(f"pdfplumber: 第 {i+1} 页提取为空")
    return "\n".join(texts)


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    """pypdf: 轻量回退"""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
        else:
            logger.debug(f"pypdf: 第 {i+1} 页提取为空")
    return "\n".join(texts)


def _extract_with_pdfminer(pdf_bytes: bytes) -> str:
    """pdfminer.six: 底层解析引擎"""
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
    )
    text = pdfminer_extract(
        io.BytesIO(pdf_bytes),
        laparams=laparams,
    )
    return text if text else ""


def _get_ocr_languages() -> str:
    """检测tesseract可用语言包"""
    import subprocess
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, timeout=5
        )
        available = result.stdout.strip().split("\n")
        langs = [l.strip() for l in available if not l.startswith("List") and l.strip()]
        selected = []
        if "chi_sim" in langs:
            selected.append("chi_sim")
        if "chi_tra" in langs:
            selected.append("chi_tra")
        selected.append("eng")
        return "+".join(selected)
    except Exception:
        return "eng"


def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """OCR路径: PDF → 图片 → tesseract识别"""
    from pdf2image import convert_from_bytes
    import pytesseract

    lang = _get_ocr_languages()
    logger.info(f"OCR使用语言: {lang}")

    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as e:
        logger.error(f"PDF→图片转换失败: {e}")
        return ""

    texts = []
    for i, img in enumerate(images):
        try:
            page_text = pytesseract.image_to_string(img, lang=lang)
            if page_text and page_text.strip():
                texts.append(page_text.strip())
            else:
                logger.debug(f"OCR: 第 {i+1} 页识别为空")
        except Exception as e:
            logger.warning(f"OCR: 第 {i+1} 页识别失败: {e}")
            continue

    return "\n".join(texts)


def _is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """启发式判断PDF是否是扫描件"""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            check_pages = min(len(pdf.pages), 5)
            text_pages = 0
            for page in pdf.pages[:check_pages]:
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    text_pages += 1
            return text_pages / check_pages < 0.2
    except Exception:
        return False


def _clean_and_truncate(text: str, max_length: int) -> str:
    """统一清理和截断"""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


@traceable(name="extract_pdf_text", run_type="tool")
def extract_pdf_text(pdf_bytes: bytes, max_length: int = 20000) -> str:
    """PDF文本提取主入口（多层回退策略）"""
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        return "提取失败: 内容不是有效的PDF文件"

    if _is_scanned_pdf(pdf_bytes):
        logger.info("检测到扫描件PDF，直接使用OCR路径")
        text = _extract_with_ocr(pdf_bytes)
        if _is_meaningful_text(text):
            return _clean_and_truncate(text, max_length)
        return "提取失败: 扫描件OCR识别未能提取出有意义的文本"

    layers = [
        ("pdfplumber", _extract_with_pdfplumber),
        ("pypdf",      _extract_with_pypdf),
        ("pdfminer",   _extract_with_pdfminer),
        ("OCR",        _extract_with_ocr),
    ]

    for name, extract_fn in layers:
        try:
            logger.info(f"尝试使用 {name} 提取...")
            text = extract_fn(pdf_bytes)
            if _is_meaningful_text(text):
                logger.info(f"✓ {name} 提取成功，{len(text)} 个字符")
                return _clean_and_truncate(text, max_length)
            else:
                logger.warning(f"✗ {name} 提取结果无意义，尝试下一层")
                continue
        except Exception as e:
            logger.warning(f"✗ {name} 抛出异常: {e}，尝试下一层")
            continue

    return "提取失败: 所有提取方法均未能获取有效文本内容"


# ==================== 网页内容抓取 ====================

@traceable(name="fetch_webpage", run_type="tool")
def fetch_webpage_content(url: str, max_length: int = 20000) -> str:
    """
    抓取网页内容（增强编码处理）。
    当内容为PDF时,自动走多层回退提取PDF文本。
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        # PDF 判断
        content_type = response.headers.get("Content-Type", "").lower()
        is_pdf = ("pdf" in content_type) or (response.content[:4] == b"%PDF")

        if is_pdf:
            return extract_pdf_text(response.content, max_length)

        # HTML 解析
        declared_encoding = response.encoding

        detected_encoding = None
        try:
            import chardet
            detected_result = chardet.detect(response.content)
            detected_encoding = detected_result['encoding']
            confidence = detected_result['confidence']
            if confidence > 0.7 and detected_encoding:
                declared_encoding = detected_encoding
        except Exception:
            pass

        encodings_to_try = [
            declared_encoding,
            detected_encoding,
            'utf-8', 'gb18030', 'gbk', 'big5', 'big5hkscs',
            'shift_jis', 'euc-kr', 'iso-8859-1', 'windows-1252'
        ]
        encodings_to_try = [e for e in dict.fromkeys(encodings_to_try) if e]

        html_content = None
        used_encoding = None

        for encoding in encodings_to_try:
            try:
                html_content = response.content.decode(encoding, errors='strict')
                used_encoding = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if html_content is None:
            html_content = response.content.decode('utf-8', errors='replace')
            used_encoding = 'utf-8 (with replacement)'

        soup = BeautifulSoup(html_content, 'html.parser', from_encoding=used_encoding)

        for script in soup(["script", "style", "meta", "link", "noscript"]):
            script.decompose()

        text = soup.get_text(separator=' ', strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        text = re.sub(r'\s+', ' ', text)

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    except requests.Timeout:
        return f"网页访问超时: {url}"
    except requests.RequestException as e:
        return f"网页访问失败: {str(e)[:100]}"
    except Exception as e:
        return f"内容解析失败: {str(e)[:100]}"


def fetch_webpage_content_alternative(url: str, max_length: int = 10000) -> str:
    """备用网页抓取方法（使用lxml解析器）"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        try:
            from lxml import html as lxml_html
            tree = lxml_html.fromstring(response.content)
            text = tree.text_content()
        except ImportError:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()

        text = re.sub(r'\s+', ' ', text.strip())

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    except Exception as e:
        return f"备用方法也失败: {str(e)[:100]}"
