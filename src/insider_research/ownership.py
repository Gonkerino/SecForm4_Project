"""Offline, loss-preserving extraction of SEC ownership submissions.

No role inference, position reconstruction, amendment reconciliation or features.
The field inventory is deliberately independent of SQL and retains every XML node.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import re
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from .config import FORMS

PARSER_VERSION = "ownership-1"
MAX_NODES = 100000
MAX_DEPTH = 64

# column -> (relative XML path, scalar type). Original representations and states
# live in ownership_fields even when a scalar cannot be converted.
REPORT_FIELDS = {
    "form_type": ("documentType", "text"),
    "schema_version": ("schemaVersion", "text"),
    "period_of_report": ("periodOfReport", "date"),
    "date_of_original_submission": ("dateOfOriginalSubmission", "date"),
    "issuer_cik": ("issuer/issuerCik", "cik"),
    "issuer_name": ("issuer/issuerName", "text"),
    "issuer_trading_symbol": ("issuer/issuerTradingSymbol", "text"),
    "issuer_foreign_trading_symbol": ("issuer/issuerForeignTradingSymbol", "text"),
    "no_securities_owned": ("noSecuritiesOwned", "bool"),
    "not_subject_to_section16": ("notSubjectToSection16", "bool"),
    "form3_holdings_reported": ("form3HoldingsReported", "bool"),
    "form4_transactions_reported": ("form4TransactionsReported", "bool"),
    "aff10b5_one": ("aff10b5One", "bool"),
    "remarks": ("remarks", "text"),
}
OWNER_FIELDS = {
    "owner_cik": ("reportingOwnerId/rptOwnerCik", "cik"),
    "owner_name": ("reportingOwnerId/rptOwnerName", "text"),
    **{column: ("reportingOwnerAddress/" + tag, kind) for column, tag, kind in (
        ("street1", "rptOwnerStreet1", "text"), ("street2", "rptOwnerStreet2", "text"),
        ("city", "rptOwnerCity", "text"), ("state", "rptOwnerState", "text"),
        ("zip_code", "rptOwnerZipCode", "text"), ("state_description", "rptOwnerStateDescription", "text"),
        ("non_us_address", "rptOwnerNonUSAddressFlag", "bool"),
        ("non_us_state_territory", "rptOwnerNonUSStateTerritory", "text"),
        ("country", "rptOwnerCountry", "text"))},
    **{column: ("reportingOwnerRelationship/" + tag, kind) for column, tag, kind in (
        ("is_director", "isDirector", "bool"), ("is_officer", "isOfficer", "bool"),
        ("is_ten_percent_owner", "isTenPercentOwner", "bool"), ("is_other", "isOther", "bool"),
        ("officer_title", "officerTitle", "text"), ("other_text", "otherText", "text"))},
}
POSITION_FIELDS = {
    "security_title": ("securityTitle", "text"),
    "transaction_form_type": ("transactionCoding/transactionFormType", "text"),
    "shares_owned_following": ("postTransactionAmounts/sharesOwnedFollowingTransaction", "numeric"),
    "value_owned_following": ("postTransactionAmounts/valueOwnedFollowingTransaction", "numeric"),
    "direct_or_indirect": ("ownershipNature/directOrIndirectOwnership", "text"),
    "nature_of_ownership": ("ownershipNature/natureOfOwnership", "text"),
}
DERIVATIVE_FIELDS = {
    "conversion_or_exercise_price": ("conversionOrExercisePrice", "numeric"),
    "exercise_date": ("exerciseDate", "date"),
    "expiration_date": ("expirationDate", "date"),
    "underlying_security_title": ("underlyingSecurity/underlyingSecurityTitle", "text"),
    "underlying_security_shares": ("underlyingSecurity/underlyingSecurityShares", "numeric"),
    "underlying_security_value": ("underlyingSecurity/underlyingSecurityValue", "numeric"),
}
TRANSACTION_FIELDS = {
    **POSITION_FIELDS, **DERIVATIVE_FIELDS,
    "transaction_date": ("transactionDate", "date"),
    "deemed_execution_date": ("deemedExecutionDate", "date"),
    "transaction_code": ("transactionCoding/transactionCode", "text"),
    "equity_swap_involved": ("transactionCoding/equitySwapInvolved", "bool"),
    "transaction_timeliness": ("transactionTimeliness", "text"),
    "transaction_shares": ("transactionAmounts/transactionShares", "numeric"),
    "transaction_price_per_share": ("transactionAmounts/transactionPricePerShare", "numeric"),
    "transaction_total_value": ("transactionAmounts/transactionTotalValue", "numeric"),
    "acquired_disposed_code": ("transactionAmounts/transactionAcquiredDisposedCode", "text"),
}
HOLDING_FIELDS = {**POSITION_FIELDS, **DERIVATIVE_FIELDS}
SIGNATURE_FIELDS = {"signature_name": ("signatureName", "text"), "signature_date": ("signatureDate", "date")}


class UnsupportedSubmission(ValueError):
    pass


@dataclass
class Extraction:
    header: dict
    documents: list[dict]
    header_fields: list[dict] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    owners: list[dict] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)
    holdings: list[dict] = field(default_factory=list)
    footnotes: list[dict] = field(default_factory=list)
    field_notes: list[dict] = field(default_factory=list)
    signatures: list[dict] = field(default_factory=list)
    fields: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)

    def issue(self, code, path, message, severity="warning"):
        self.issues.append(dict(code=code, source_path=path, message=message, severity=severity))

    @property
    def status(self):
        if any(i["severity"] == "review" for i in self.issues):
            return "needs_review"
        return "complete_with_warnings" if self.issues else "complete"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _text(element):
    return "".join(element.itertext()) if element is not None else None


def parse_submission(body: bytes, accession: str) -> Extraction:
    """Extract one original SGML submission. No I/O or database access."""
    if len(body) > 128 * 1024 * 1024:
        raise ValueError("Submission exceeds parser size limit")
    if not body.rstrip().upper().endswith(b"</SEC-DOCUMENT>"):
        raise ValueError("Submission has no final SEC-DOCUMENT boundary")
    header_match = re.search(rb"<SEC-HEADER>(.*?)</SEC-HEADER>", body, re.S | re.I)
    if not header_match:
        raise ValueError("Missing SEC header")
    raw_header = header_match.group(1).decode("utf-8", errors="replace")

    def header_value(label):
        match = re.search(r"^" + re.escape(label) + r":\s*([^\r\n]*)", raw_header, re.M)
        return match.group(1).strip() if match else None

    if header_value("ACCESSION NUMBER") != accession:
        raise ValueError("Submission header accession differs from registered accession")
    acceptance = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", raw_header)
    header = dict(raw_header=raw_header, accession=accession,
                  form_type=header_value("CONFORMED SUBMISSION TYPE"),
                  acceptance_raw=acceptance.group(1) if acceptance else None,
                  acceptance_at=None, acceptance_method="SEC header; America/New_York interpretation",
                  filed_date=None, period_of_report=None,
                  public_document_count=None)
    documents, payloads = [], []
    for number, match in enumerate(re.finditer(rb"(?im)^<DOCUMENT>\s*\r?\n(.*?)^</DOCUMENT>\s*$", body, re.S), 1):
        document = match.group(1)
        text_match = re.search(rb"(?im)^<TEXT>[^\S\r\n]*\r?\n", document)
        metadata = document[:text_match.start()] if text_match else document

        def document_value(tag):
            m = re.search(rb"(?im)^<" + tag.encode() + rb">([^\r\n]*)", metadata)
            return m.group(1).decode("utf-8", errors="replace").strip() if m else None

        if text_match:
            end = document.upper().rfind(b"</TEXT>")
            if end < text_match.end():
                raise ValueError("Embedded document has no closing TEXT boundary")
            payload = document[text_match.end():end]
            offset = match.start(1) + text_match.end()
        else:
            payload, offset = b"", match.start(1) + len(document)
        documents.append(dict(document_number=number, sequence=document_value("SEQUENCE"),
                              document_type=document_value("TYPE"), filename=document_value("FILENAME"),
                              description=document_value("DESCRIPTION"), byte_offset=offset,
                              byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                              raw_metadata=metadata.decode("utf-8", errors="replace")))
        payloads.append(payload)
    if not documents:
        raise UnsupportedSubmission("No supported embedded document boundaries")
    result = Extraction(header, documents)
    # The SEC header also carries issuer/owner sections, file numbers, historical
    # names and other metadata beyond the XML. Preserve queryable line evidence
    # with repeated-section identities instead of guessing one flat entity record.
    sections, section_counts = [], Counter()
    for line_number, line in enumerate(raw_header.splitlines(), 1):
        if not line.strip():
            continue
        indentation = len(line.expandtabs(8)) - len(line.expandtabs(8).lstrip())
        while sections and sections[-1][0] >= indentation:
            sections.pop()
        section_path = sections[-1][1] if sections else ""
        match = re.match(r"\s*([^:<]+):[ \t]*(.*)", line)
        tag_match = re.match(r"\s*<([^>]+)>(.*)", line)
        key, value = (match.group(1).strip(), match.group(2)) if match else (tag_match.group(1), tag_match.group(2)) if tag_match else (None, line.strip())
        result.header_fields.append(dict(line_number=line_number, section_path=section_path,
                                         field_name=key, value=value, raw_line=line))
        if match and not value.strip():
            section_counts[(section_path, key)] += 1
            sections.append((indentation, f"{section_path}/{key}[{section_counts[(section_path, key)]}]"))
    if "\ufffd" in raw_header:
        result.issue("header_decoding", "SEC-HEADER", "Header contains replacement characters; original bytes retained", "review")
    for key, label in (("filed_date", "FILED AS OF DATE"), ("period_of_report", "CONFORMED PERIOD OF REPORT")):
        value = header_value(label)
        if value:
            try:
                header[key] = datetime.strptime(value, "%Y%m%d").date()
            except ValueError:
                result.issue("invalid_header_date", label, f"Invalid header date: {value}", "review")
    if acceptance:
        try:
            naive = datetime.strptime(acceptance.group(1), "%Y%m%d%H%M%S")
            header["acceptance_at"] = naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
        except ValueError:
            result.issue("invalid_acceptance", "SEC-HEADER", "Invalid acceptance timestamp", "review")
    else:
        result.issue("missing_acceptance", "SEC-HEADER", "No acceptance timestamp was supplied")
    count = header_value("PUBLIC DOCUMENT COUNT")
    if count and count.isdigit():
        header["public_document_count"] = int(count)
        if int(count) != len(documents):
            result.issue("document_count_mismatch", "SEC-HEADER", "Document inventory differs from header", "review")
    elif count:
        result.issue("invalid_document_count", "SEC-HEADER", "Non-numeric public document count", "review")
    candidates = [i for i, doc in enumerate(documents) if doc["document_type"] in FORMS]
    if not candidates:
        raise UnsupportedSubmission("No ownership primary document; legacy non-XML submissions require review")
    if len(candidates) != 1:
        raise UnsupportedSubmission("Ambiguous submission: multiple ownership primary documents")
    primary = candidates[0]
    result.report["document_number"] = primary + 1
    xml = payloads[primary].strip()
    xml = re.sub(rb"^<XML>\s*", b"", xml, flags=re.I)
    xml = re.sub(rb"\s*</XML>$", b"", xml, flags=re.I)
    # Reject DTD/entity declarations before parsing (including NUL-separated UTF-16).
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", xml.replace(b"\x00", b""), re.I):
        raise ValueError("DTD and entity declarations are prohibited")
    try:
        pull = ET.XMLPullParser(events=("start", "end"))
        root, depth, node_count = None, 0, 0
        for offset in range(0, len(xml), 65536):
            pull.feed(xml[offset:offset + 65536])
            for event, node in pull.read_events():
                if event == "start":
                    if root is None:
                        root = node
                    depth += 1
                    node_count += 1
                    if depth > MAX_DEPTH or node_count > MAX_NODES:
                        raise ValueError("Ownership XML exceeds depth/node limits")
                else:
                    depth -= 1
        pull.close()
    except (ET.ParseError, LookupError) as exc:
        raise ValueError(f"Invalid ownership XML: {exc}") from exc
    if root is None or _local(root.tag) != "ownershipDocument":
        raise UnsupportedSubmission("Primary document is not ownershipDocument XML")
    # Index every node first, preserving namespace, attributes, text, tail and order.
    paths, nodes, child_index = {}, [], {}
    stack = [(root, "/ownershipDocument[1]", 1)]
    while stack:
        node, path, depth = stack.pop()
        if depth > MAX_DEPTH or len(nodes) >= MAX_NODES:
            raise ValueError("Ownership XML exceeds depth/node limits")
        if len(path.encode("utf-8")) > 2000:
            raise ValueError("Ownership XML source path exceeds storage limit")
        paths[id(node)] = path
        nodes.append(node)
        counts, children, grouped = Counter(), [], {}
        for child in node:
            name = _local(child.tag)
            counts[name] += 1
            grouped.setdefault(name, []).append(child)
            children.append((child, f"{path}/{name}[{counts[name]}]", depth + 1))
        child_index[id(node)] = grouped
        stack.extend(reversed(children))

    consumed = set()

    def find_all(base, relative):
        found = [base]
        for segment in relative.split("/"):
            found = [child for parent in found for child in child_index[id(parent)].get(segment, ())]
        return found

    def consume_ancestors(base, relative):
        # Traverse once, preserving all matching ancestors (including repeated
        # and foreign-namespace elements) exactly as the original extraction did.
        found = [base]
        for segment in relative.split("/"):
            found = [child for parent in found for child in child_index[id(parent)].get(segment, ())]
            consumed.update(id(n) for n in found)
        return found

    def scalar(base, relative, kind):
        matches = consume_ancestors(base, relative)
        if not matches:
            return None
        if len(matches) > 1:
            result.issue("repeated_scalar", paths[id(base)], f"Repeated field {relative}", "review")
        node = matches[0]
        values = [c for c in node if _local(c.tag) == "value"]
        if values:
            consumed.update(id(v) for v in values)
            if len(values) > 1:
                result.issue("repeated_value", paths[id(node)], "Multiple scalar value elements", "review")
            raw = _text(values[0])
        else:
            raw = node.text
        value = raw.strip() if raw else ""
        if not value:
            return None
        try:
            if kind == "date":
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError()
                return date.fromisoformat(value)
            if kind == "numeric":
                if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value) or len(value) > 1000:
                    raise ValueError()
                parsed = Decimal(value)
                if not parsed.is_finite():
                    raise ValueError()
                return parsed
            if kind == "bool":
                if value not in ("0", "1", "true", "false"):
                    raise ValueError()
                return value in ("1", "true")
            if kind == "cik" and not re.fullmatch(r"\d{1,10}", value):
                raise ValueError()
            return value
        except (ValueError, InvalidOperation):
            result.issue("invalid_scalar", paths[id(node)], f"Invalid {kind} representation: {value[:200]}", "review")
            return None

    def extract(base, mapping):
        consumed.add(id(base))
        return {column: scalar(base, path, kind) for column, (path, kind) in mapping.items()}

    result.report.update(extract(root, REPORT_FIELDS))
    if result.report["form_type"] not in FORMS:
        raise ValueError("Unsupported or missing ownership documentType")
    for label, form in (("SEC header", header["form_type"]), ("document inventory", documents[primary]["document_type"])):
        if form != result.report["form_type"]:
            result.issue("form_mismatch", label, "Form differs from ownership XML", "review")
    if header["period_of_report"] != result.report["period_of_report"]:
        result.issue("report_date_mismatch", "periodOfReport", "Header and XML report dates differ")
    for key in ("issuer_cik", "issuer_name", "period_of_report"):
        if result.report[key] is None:
            result.issue("missing_required", paths[id(root)], f"Missing report field {key}", "review")
    for owner in find_all(root, "reportingOwner"):
        row = dict(row_number=len(result.owners) + 1, source_path=paths[id(owner)], **extract(owner, OWNER_FIELDS))
        result.owners.append(row)
        if row["owner_cik"] is None or row["owner_name"] is None:
            result.issue("missing_owner_identity", row["source_path"], "Missing owner CIK/name", "review")
        if row["is_officer"] is True and not row["officer_title"]:
            result.issue("missing_officer_title", row["source_path"], "Officer flag supplied without a title", "review")
    if not result.owners:
        result.issue("missing_owner", paths[id(root)], "No reporting owners", "review")
    for table_name, derivative in (("nonDerivativeTable", False), ("derivativeTable", True)):
        for table in find_all(root, table_name):
            consumed.add(id(table))
            for node in table:
                tag = _local(node.tag)
                prefix = "derivative" if derivative else "nonDerivative"
                if tag not in (prefix + "Transaction", prefix + "Holding"):
                    continue
                transactions = tag.endswith("Transaction")
                mapping = TRANSACTION_FIELDS if transactions else HOLDING_FIELDS
                target = result.transactions if transactions else result.holdings
                row = dict(row_number=len(target) + 1, is_derivative=derivative,
                           source_path=paths[id(node)], **extract(node, mapping))
                target.append(row)
                for key in ("security_title", "direct_or_indirect"):
                    if row[key] is None:
                        result.issue("missing_row_field", row["source_path"], f"Missing {key}", "review")
                for key, allowed in (("direct_or_indirect", {"D", "I"}), ("acquired_disposed_code", {"A", "D"})):
                    if row.get(key) is not None and row[key] not in allowed:
                        result.issue("invalid_code", row["source_path"], f"Unexpected {key}: {row[key]}", "review")
                if transactions and row["transaction_code"] is not None and row["transaction_code"] not in set("ACDEFGHIJLMOPSUWXZ"):
                    result.issue("unknown_transaction_code", row["source_path"], "Unrecognised transaction code", "review")
                if transactions:
                    for required in ("transactionDate", "transactionCoding/transactionCode",
                                     "transactionAmounts/transactionAcquiredDisposedCode"):
                        if not find_all(node, required):
                            result.issue("missing_transaction_field", row["source_path"], f"Missing {required}", "review")
                if row.get("transaction_timeliness") not in (None, "E", "L"):
                    result.issue("unknown_timeliness", row["source_path"], "Unexpected timeliness code", "review")
    for container in find_all(root, "footnotes"):
        consumed.add(id(container))
        for node in container:
            if _local(node.tag) != "footnote":
                continue
            consumed.add(id(node))
            result.footnotes.append(dict(row_number=len(result.footnotes) + 1,
                                         source_path=paths[id(node)], footnote_id=node.get("id"), text=_text(node)))
    note_counts = Counter(n["footnote_id"] for n in result.footnotes)
    if None in note_counts or any(count > 1 for count in note_counts.values()):
        result.issue("invalid_footnote_ids", "footnotes", "Missing or duplicate footnote IDs", "review")
    for node in find_all(root, "ownerSignature"):
        signature = dict(row_number=len(result.signatures) + 1,
                         source_path=paths[id(node)], **extract(node, SIGNATURE_FIELDS))
        result.signatures.append(signature)
        if signature["signature_name"] is None or signature["signature_date"] is None:
            result.issue("missing_signature_field", signature["source_path"], "Missing signature name/date", "review")
    if not result.signatures:
        result.issue("missing_signature", "ownerSignature", "No signature supplied", "review")
    for node in nodes:
        path = paths[id(node)]
        tag = _local(node.tag)
        namespace = node.tag.rsplit("}", 1)[0] if "}" in node.tag else ""
        root_namespace = root.tag.rsplit("}", 1)[0] if "}" in root.tag else ""
        if namespace != root_namespace:
            result.issue("foreign_namespace", path, "Element namespace differs from document namespace", "review")
        if tag == "footnoteId":
            consumed.add(id(node))
            ref = node.get("id")
            result.field_notes.append(dict(source_path=path, field_path=path.rsplit("/", 1)[0], footnote_id=ref))
            if note_counts.get(ref, 0) != 1:
                result.issue("unresolved_footnote", path, f"Footnote reference {ref!r} is missing or ambiguous", "review")
        unknown = id(node) not in consumed
        if unknown:
            result.issue("unmapped_element", path, f"Preserved but not mapped: {tag}", "review")
        allowed_attrs = {"id"} if tag in ("footnote", "footnoteId") else set()
        allowed_attrs |= {"{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
                          "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"}
        if set(node.attrib) - allowed_attrs:
            result.issue("unmapped_attribute", path, "Unexpected attributes preserved", "review")
        value_nodes = [c for c in node if _local(c.tag) == "value"]
        has_notes = any(_local(c.tag) == "footnoteId" for c in node)
        if value_nodes:
            raw = _text(value_nodes[0])
            state = "value" if raw.strip() else "empty"
        elif not len(node):
            raw = node.text or ""
            state = "value" if raw.strip() else "empty"
        elif has_notes and not (node.text or "").strip():
            state = "footnote_only"
        else:
            state = "container"
        result.fields.append(dict(source_path=path, tag=node.tag, text=node.text,
                                  tail=node.tail, attributes=dict(node.attrib), value_state=state,
                                  mapped=not unknown))
    if result.report["no_securities_owned"] and (result.transactions or result.holdings):
        result.issue("conflicting_no_securities", "noSecuritiesOwned", "Flag conflicts with reported rows")
    if result.report["form_type"] in ("3", "3/A") and result.transactions:
        result.issue("form3_transactions", "documentType", "Form 3 unexpectedly contains transaction rows", "review")
    if result.report["aff10b5_one"] is False and any(re.search(r"10b5[\s-]*1", n["text"] or "", re.I) for n in result.footnotes):
        result.issue("plan_evidence_conflict", "aff10b5One", "False indicator with 10b5-1 footnote; both preserved")
    return result
