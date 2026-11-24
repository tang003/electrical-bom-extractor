"""Shared IEC/EPLAN identifiers and catalogue tokens, independent of customer names."""
from __future__ import annotations
import re

_CONTEXT = r"(?:=[A-Z0-9_.]+)?(?:\+[A-Z0-9_.]+)?"
_LOCAL = r"-\d*[A-Z]{1,5}\d+(?:[.-]\d+)*[A-Z]?"
COMPONENT_TAG_RE = re.compile(rf"^(?:{_CONTEXT}(?:{_LOCAL}|-(?:KA|SQ|FAN|PLC|CPU|WL))|(?:FR|HL)\d+|SQ(?:-1|\d+)?)$", re.I)
CAD_UNPREFIXED_TAG_RE = re.compile(r"^(?:QF|QS|CP|FU|FR|KM|KA|KT|KV|K|SB|SA|HL|HG|HR|HA|H|WL|W|XT|X|TC|TM|PS|PLC|CPU|NFB|MCCB|MCB|MC|OCR|INV|CR|PB|PBL|PL|SCR|SV)\d+(?:[.-]\d+)*[A-Z]?$", re.I)
# Bare labels in printed schematics also occur on IO cards and terminal legends.
# Limit implicit PDF anchors to device designations with a distinct electrical role.
PDF_DEVICE_RE = re.compile(r'(?:QF|QS|FU|KM|KA|KT|KV|MC|TH|NFB|OCR|MCCB|MCB|SB|SA|PBL|SCR|INV)\d+(?:[.-]\d+)*[A-Z]?',re.I)
_PREFIXED_ANCHOR_RE=re.compile(r'(?<![A-Za-z0-9])'+COMPONENT_TAG_RE.pattern[1:-1]+r'(?=$|[\s,，、;；])',re.I)
_CAD_ANCHOR_RE=re.compile(r'(?<![A-Za-z0-9-])'+CAD_UNPREFIXED_TAG_RE.pattern[1:-1]+r'(?=$|\s)',re.I)
_LOCAL_SEARCH = re.compile(r"-\d*[A-Z][A-Z0-9]*(?:[.-]\d+)*$", re.I)


def local_tag(tag: str) -> str:
    value = tag.strip().upper()
    match = _LOCAL_SEARCH.search(value)
    return match.group(0) if match else value


def device_code(tag: str) -> str:
    match = re.match(r"\d*([A-Z]+)", local_tag(tag).lstrip("-"))
    return match.group(1) if match else ""


def qualify_tag(tag: str, location: str = "", function: str = "") -> str:
    value = tag.strip().upper()
    if value.startswith("=") or not (location or function):
        return value
    if value.startswith("+"):
        return function + value
    return function + location + (value if value.startswith("-") else "-" + value)


def with_component_anchors(spans):
    """Keep literal spans and add bounded tag anchors for combined/native CAD text.

    No grouped-tag expansion or invented suffixes: QF0 400A 3P yields a QF0
    anchor, while M16, AC380V, and a grouped QF1,2 do not become new devices.
    """
    result=[]
    for span in spans:
        result.append(span)
        text=span['text'].strip()
        if COMPONENT_TAG_RE.fullmatch(text):continue
        matches=list(_PREFIXED_ANCHOR_RE.finditer(text))
        if span.get('source')=='cad_native':
            matches+=list(_CAD_ANCHOR_RE.finditer(text))
            if text.upper()=='WL':matches+=list(re.finditer(r'^WL$',text,re.I))
        else:
            bare=PDF_DEVICE_RE.match(text)
            if bare and (bare.end()==len(text) or text[bare.end()].isspace()):
                matches.append(bare)
        for match in matches:
            if re.match(r'[,，、~～]\s*\d',text[match.end():]):continue
            literal=match.group(0);tag=literal if literal.startswith(('-', '=', '+')) else '-'+literal
            x0,y0,x1,y1=span['bbox']
            # Partial horizontal text spans need a tag-sized anchor, not the whole spec line.
            if len(literal)!=len(text) and x1-x0>y1-y0:
                width=x1-x0;x1=x0+width*match.end()/len(text);x0+=width*match.start()/len(text)
            box=[x0,y0,x1,y1]
            n0,n1,n2,n3=span.get('norm_bbox',[0,0,0,0])
            if span['bbox'][2]>span['bbox'][0]:
                nx0=n0+(n2-n0)*(x0-span['bbox'][0])/(span['bbox'][2]-span['bbox'][0])
                nx1=n0+(n2-n0)*(x1-span['bbox'][0])/(span['bbox'][2]-span['bbox'][0])
            else:nx0,nx1=n0,n2
            result.append({**span,'text':tag,'source_text':text,'native_tag':literal,'unindexed_tag':literal.upper()=='WL','bbox':box,
                           'norm_bbox':[nx0,n1,nx1,n3],'cx':(x0+x1)/2,'cy':(y0+y1)/2})
    return result


def is_connection_tag(tag: str) -> bool:
    code = device_code(tag)
    # A printed -L1 can be a transformer, -X1 a socket/terminal, -W1 a cable.
    # Preserve those material candidates; bare phase names are not COMPONENT_TAGs.
    # V2-07 / V2+01 are potential names on off-page arrows, not V devices.
    return code in {"N", "PE", "PEN"} or bool(
        re.fullmatch(r"-V\d+[+-]\d+[A-Z]?", local_tag(tag), re.I)
    )


def is_field_load(tag: str) -> bool:
    return device_code(tag) in {"M", "Y", "YV", "DT", "SP"}


_POWER_CURRENT = r"\d+(?:\.\d+)?\s*(?:KW|KVA|HP|A)"
POWER_CURRENT_LABEL = re.compile(rf"(?=.*(?:KW|KVA|HP))(?=.*\d\s*A\b){_POWER_CURRENT}(?:\s*[,，;；/]\s*|\s+){_POWER_CURRENT}", re.I)


def tagged_model_series(tag, model):
    """A device may use its catalogue family followed by an installation index."""
    head = re.fullmatch(r'-([A-Z]{1,8}\d{2,})-\d+',local_tag(tag),re.I)
    return bool(head and model.upper().startswith(head[1]+'-'))


MODEL_TOKEN_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9]*[-/][A-Z0-9][A-Z0-9+./-]*|"
    r"(?:6ES7|6EP|6AV|3RT|3RV|5SU|5SY)\s*[A-Z0-9][A-Z0-9+./ -]*|"
    r"[A-Z]{2,}\d[A-Z0-9]*|(?:IC65N|C65|GV2|EZD|LC1|LRD|A9|RM\d|RXM|REXL|AMC|ACS|EDS|FBOX|NHR|XB2|ZB2)[A-Z0-9+./ -]+)$", re.I)
_VALUE = re.compile(
    r"^(?:(?:AC|DC)?\d+(?:\.\d+)?(?:V|A|W|KW|HZ|MA|MM|MM2|P)(?:AC|DC|/\d+P)?|"
    r"(?:AC|DC)\d+[VA](?:\d+P)?|RS\d+|RO\d+(?:[ABC]|NO|NC|COM)|"
    r"(?:DI|DO|DQ|AI|AO|PE|GND|COM|DCOM|OPN|SHUT)\d*|[PN]\d+V\d+|[EAIQMX]\d+\.\d+)$", re.I)


def is_electrical_parameter(text: str) -> bool:
    return bool(_VALUE.fullmatch(text.replace(' ','')) or re.fullmatch(r'[BCD]\d+A?/\d+P',text,re.I))


def is_connection_label(text: str) -> bool:
    """Recognize wire and signal identifiers that resemble catalogue numbers."""
    value = text.strip().upper()
    return bool(
        re.fullmatch(r'[A-Z]{1,5}\d{1,3}(?:L[123]|PE|_[A-Z0-9()+-]+)', value)
        or re.fullmatch(r'V\d+[+-]\d+[A-Z]?', value)
        or re.fullmatch(r'(?:V|I|Q|M|G|DI|DO|AI|AO)_[A-Z0-9_]+', value)
        or re.fullmatch(r'RJ\d{2,3}', value)
        or re.fullmatch(r'(?:CH|IN|OUT|FAULT|ERR|ALM|READY|FLAG)\d{1,3}', value)
    )


def extract_model_strings(text: str) -> list[str]:
    """Read embedded part numbers, rejecting electrical values, signals and tags."""
    text = text.strip().replace("−", "-").replace("–", "-")
    if not text or text.startswith(("-", "=", "+")) or COMPONENT_TAG_RE.fullmatch(text) or is_electrical_parameter(text) or is_connection_label(text):
        return []
    # A catalogue prefix plus the device designation is one printed order code.
    # Splitting CK3W_AX2323N into CK3W and AX2323N invents a model conflict.
    if re.fullmatch(r'[A-Z]{2,}[A-Z0-9]*_[A-Z]{1,6}\d+[A-Z0-9]*', text, re.I):
        return [text]
    if model_family(text):
        return [text]
    if re.match(r"^(?:I?C65N?|GV2|EZD|LC1|LRD|XB2|ZB2|RXM|REXL|NHR)[A-Z0-9-]", text, re.I) and re.fullmatch(r"[A-Za-z0-9+./ *-]+", text):
        return [text]
    matches = []
    # The base of a Siemens MLFB may be printed in two groups separated by
    # whitespace. Consume the second group only when it leads into its suffix.
    for match in re.finditer(r"(?<![A-Za-z0-9])(?:6ES7|6EP|6AV|3RT|3RV|5SU|5SY)"
                             r"[0-9A-Z]*(?:\s+\d[0-9A-Z]*(?=-))?(?:-[0-9A-Z]+)+", text, re.I):
        matches.append(re.sub(r"\s+", "", match.group(0)))
    if not matches:
        for match in re.finditer(r"(?<![A-Za-z0-9])(?:6ES7\s*\d[0-9A-Z-]+|(?:6EP|6AV|3RT|3RV|5SU|5SY)[0-9A-Z-]+)", text, re.I):
            matches.append(re.sub(r"\s+", "", match.group(0)))
    if matches:
        return list(dict.fromkeys(matches))
    # Decimal size codes are part of many drive, fuse and pneumatic part numbers.
    # A parenthesized capacity immediately after the identifier is also printed evidence.
    segment = r"[A-Za-z0-9]+(?:\.\d+[A-Za-z0-9]*)*"
    capacity = r"(?:[(（]\d+(?:\.\d+)?\s*(?:[kK](?:[wW]|VA)?|HP|A|V)[)）])?"
    for match in re.finditer(
        rf"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*(?:[-/]{segment})+(?:\+{segment}(?:[-/]{segment})*)*{capacity})",
        text,
    ):
        value = match.group(1)
        if len(value) >= 5 and re.search(r"\d", value) and not _VALUE.fullmatch(value):
            matches.append(value)
    if not matches:
        for value in re.findall(r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9]{4,}|[A-Z]{2,}\d[A-Z0-9]*)(?![A-Za-z0-9])", text):
            if re.search(r"\d", value) and not _VALUE.fullmatch(value):
                matches.append(value)
    if not matches and model_family(text):
        matches.append(text)
    return list(dict.fromkeys(value for value in matches if not CAD_UNPREFIXED_TAG_RE.fullmatch(value)
                              and not PDF_DEVICE_RE.fullmatch(value) and not is_connection_label(value)))


def model_family(model: str) -> bool:
    return bool(re.fullmatch(r"(?:I?C65N?|SMART\s+\d+\s+IE(?:\s+V\d+)?|ACS\d+\s+\d+\s*KW)", model, re.I))


BREAKER_FAMILIES = ("EZD", "NSX", "GV2", "IC65", "C65", "A9D")


def model_compatible(tag: str, model: str) -> bool:
    code, value = device_code(tag), model.upper().replace(" ", "")
    tagged_code = re.fullmatch(r'[A-Z]{2,}[A-Z0-9]*_([A-Z]{1,6}\d+[A-Z0-9]*)', value)
    if tagged_code:
        return local_tag(tag).lstrip('-') == tagged_code[1]
    families = [
        (("6AV", "SMART"), {"HMI", "A", "U", "PLC", "CPU", "AI", "AO", "DI", "DO"}),
        (("6ES7", "EDS-", "FBOX-", "SXC-"), {"A", "U", "PLC", "CPU", "AI", "AO", "DI", "DO"}),
        (("EDR-", "NDR-", "HDR-", "LRS-", "DR-", "6EP"), {"G", "GS", "PS", "U", "A", "V","DCP"}),
        (("ACS",), {"TB", "A", "U", "V", "G"}),
        (("AMC", "PZ", "PD"), {"P", "PM", "PA", "A", "U"}),
        (("LC1",), {"KM", "MC", "K", "Q"}), (("LRD","LRN"), {"FR","TH","OCR","F"}),
        (BREAKER_FAMILIES, {"QF", "QS", "Q", "F"}),
        (("RM22",), {"KPH", "K", "KV"}), (("REXL", "RXM"), {"KT", "KA", "K"}),
        (("XB2BD",), {"SA", "S", "Q"}),
    ]
    for prefixes, codes in families:
        if value.startswith(prefixes):
            return code in codes
    return not is_connection_tag(tag) and not is_field_load(tag) and (code=='WL' or not code.startswith(("X", "W")))
