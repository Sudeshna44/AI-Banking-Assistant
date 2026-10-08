from types import SimpleNamespace
import unittest

from banking_assistant import (
    ANSWER_NOT_FOUND,
    format_retrieved_context,
    get_source_references,
)


class BankingAssistantTests(unittest.TestCase):
    def test_source_references_use_one_based_pages_and_deduplicate(self):
        documents = [
            SimpleNamespace(metadata={"source": "home_loan.pdf", "page": 2}),
            SimpleNamespace(metadata={"source": "home_loan.pdf", "page": 2}),
            SimpleNamespace(metadata={"source": "faq.pdf", "page": 0}),
        ]

        self.assertEqual(
            get_source_references(documents),
            ["home_loan.pdf — Page 3", "faq.pdf — Page 1"],
        )

    def test_source_reference_without_page_uses_filename(self):
        document = SimpleNamespace(metadata={"source": "/tmp/debit_card_policy.pdf"})

        self.assertEqual(
            get_source_references([document]), ["debit_card_policy.pdf"]
        )

    def test_not_found_response_is_explicit(self):
        self.assertEqual(
            ANSWER_NOT_FOUND,
            "I don't have enough information in the bank documents to answer this question.",
        )

    def test_retrieved_context_includes_page_reference_and_text(self):
        document = SimpleNamespace(
            metadata={"source": "home_loan.pdf", "page": 2},
            page_content="Income documents are required.",
        )

        self.assertEqual(
            format_retrieved_context([document]),
            "[Source: home_loan.pdf, page 3]\nIncome documents are required.",
        )
