import pymupdf
import pytest

from backend import document_processor as dp


def make_pdf(text="Jane Doe\njane@example.com\nSkills: Python, SQL", password=None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    if password:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        data = doc.tobytes()
    doc.close()
    return data


def test_valid_txt():
    doc = dp.process_upload("resume.txt", "Jane Doe\r\n\r\n\r\n\r\nPython".encode())
    assert doc.file_type == "TXT" and doc.text == "Jane Doe\n\nPython"
    assert doc.char_count == len(doc.text) and doc.page_count is None


def test_txt_utf8_bom_and_unicode():
    doc = dp.process_upload("r.txt", "\ufeffஆசிரியர் Café".encode("utf-8"))
    assert "Café" in doc.text and not doc.text.startswith("\ufeff")


def test_txt_non_utf8_falls_back_with_warning():
    doc = dp.process_upload("r.txt", "Café résumé".encode("latin-1"))
    assert any("UTF-8" in w for w in doc.warnings)


@pytest.mark.parametrize("data", [b"", b"   \n\t  "])
def test_empty_txt_rejected(data):
    with pytest.raises(dp.DocumentError):
        dp.process_upload("empty.txt", data)


def test_binary_txt_rejected():
    with pytest.raises(dp.DocumentError, match="binary"):
        dp.process_upload("x.txt", b"abc\x00\x01\x02def")


@pytest.mark.parametrize("name", ["resume.docx", "photo.png", "noext", "script.exe"])
def test_unsupported_types(name):
    with pytest.raises(dp.DocumentError, match="Unsupported"):
        dp.process_upload(name, b"hello world")


def test_oversize_rejected(monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "0.001")
    with pytest.raises(dp.DocumentError, match="exceeds"):
        dp.process_upload("big.txt", b"a" * 5000)


def test_valid_pdf():
    doc = dp.process_upload("resume.pdf", make_pdf())
    assert doc.file_type == "PDF" and doc.page_count == 1 and "jane@example.com" in doc.text


def test_blank_pdf_reports_scanned():
    with pytest.raises(dp.DocumentError, match="scanned"):
        dp.process_upload("blank.pdf", make_pdf(text=""))


def test_corrupted_pdf():
    with pytest.raises(dp.DocumentError):
        dp.process_upload("bad.pdf", b"%PDF-1.4 this is not really a pdf")


def test_fake_pdf_extension():
    with pytest.raises(dp.DocumentError, match="not a valid PDF"):
        dp.process_upload("fake.pdf", b"just plain text")


def test_encrypted_pdf():
    with pytest.raises(dp.DocumentError, match="password"):
        dp.process_upload("locked.pdf", make_pdf(password="secret"))


def test_all_samples_load():
    samples = dp.list_samples()
    assert len(samples) >= 3
    for path in samples:
        doc = dp.load_sample(path)
        assert doc.is_sample and doc.char_count > 0


def test_truncate():
    text, cut = dp.truncate_for_model("abcdef", 3)
    assert text == "abc" and cut
    assert dp.truncate_for_model("abc", 10) == ("abc", False)
