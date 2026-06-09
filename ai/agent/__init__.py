"""Answer strategies for the AI layer.

An ``AnswerStrategy`` turns a natural-language question into a grounded answer by
routing it to the analytics/retrieval tools and formatting the result with source
citations. W3.2 ships the deterministic rule-based baseline; W3.3 adds further
strategies behind the same interface for comparison.
"""
