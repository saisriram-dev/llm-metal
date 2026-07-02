# Standard library imports
import logging
import os

# Local imports
import gst_basemodel

# Google Gemini API client, configuration types, and typed error hierarchy
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Loads environment variables from .env file for secure API key management
from dotenv import load_dotenv

# Exception class for handling Pydantic validation errors
from pydantic import ValidationError

# Retry/backoff toolkit for the network-level failure modes (rate limits, 5xx, timeouts)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gst_extractor")

# Load environment variables from .env file
# This allows us to keep the API key out of source code
load_dotenv()

# Initialize Gemini API client with the API key from environment variable
# The key "llm-metal-gemini" should be set in your .env file
client = genai.Client(api_key=os.getenv("llm-metal-gemini"))

# System instruction tells Gemini how to behave when extracting data
SYSTEM_INSTRUCTION = (
    "You extract structured GST invoice data from messy, unstructured text "
    "(OCR output, pasted emails, scanned-invoice dumps, etc.). "
    "Always return your best-effort extraction even if some fields are unclear -- "
    "use sensible defaults (e.g. empty string) rather than omitting a field. "
    "Dates must be in YYYY-MM-DD format. "
    "'amount' on the top-level object is the total invoice amount. "
    "Each line item's 'amount' is quantity * unit_price for that line."
)

# Per-request network timeout (milliseconds), passed to the Gemini client below.
REQUEST_TIMEOUT_MS = 30_000


# --- Custom exceptions for failure modes the Gemini API doesn't raise cleanly ---

class SafetyBlockedError(Exception):
    """Raised when Gemini blocks the prompt or response due to safety filters."""


class EmptyResponseError(Exception):
    """Raised when Gemini returns no text at all (transient/backend hiccup)."""


class PartialResponseError(Exception):
    """Raised when generation was cut off before completion (e.g. hit max output tokens)."""


# Sample messy invoice text for testing
messy_text = """
Invoice Receipt
Vendor: TechCorp Solutions Pvt Ltd
Date of Issue: 2023-10-15
GSTIN: 29ABCDE1234F1Z5
Total Amount: 12500.00

Items purchased:
1. Premium Widgets - 2.0 qty @ 5000.00/unit = 10000.00
2. Express Shipping - 1.0 qty @ 2500.00/unit = 2500.00
"""


def _is_retryable(exc: BaseException) -> bool:
    """
    Decides which failures are worth an exponential-backoff retry at the
    transport layer, vs. failures that need a different prompt (handled by
    the outer validation-retry loop in `extractor`).

    Retryable:
      - 429 (rate limit -- requests-per-minute / per-day on free tier)
      - any 5xx server error
      - request timeouts
      - empty response bodies (usually a transient backend glitch)

    NOT retryable here (won't be fixed by waiting and resending the same prompt):
      - 4xx client errors other than 429 (bad request, auth, etc.)
      - safety blocks (need a reworded/redacted prompt, not a delay)
      - schema validation failures / malformed JSON (handled by outer loop)
    """
    if isinstance(exc, genai_errors.ClientError):
        return exc.code == 429
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, (TimeoutError, EmptyResponseError)):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini(prompt: str, config: types.GenerateContentConfig) -> str:
    """
    Single hardened call to Gemini. Wrapped with tenacity so 429s / 5xx /
    timeouts get retried with exponential backoff, without polluting the
    schema-repair retry loop in `extractor`.

    Raises SafetyBlockedError / PartialResponseError / EmptyResponseError on
    non-network failure modes, all left for the caller to handle explicitly
    rather than silently swallowed.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            logger.warning("Rate limited (429) by Gemini API -- backing off: %s", exc)
        else:
            logger.error("Non-retryable client error (code=%s): %s", exc.code, exc)
        raise
    except genai_errors.ServerError as exc:
        logger.warning("Gemini server error (code=%s) -- will retry: %s", exc.code, exc)
        raise
    except TimeoutError as exc:
        logger.warning("Gemini request timed out -- will retry: %s", exc)
        raise

    # --- Safety-blocked responses ---
    # A blocked prompt shows up on prompt_feedback; a blocked completion shows
    # up as a finish_reason on the (possibly absent) candidate.
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    if block_reason:
        raise SafetyBlockedError(f"Prompt blocked by Gemini safety filters: {block_reason}")

    candidates = getattr(response, "candidates", None) or []
    finish_reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
    if finish_reason.upper() in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"):
        raise SafetyBlockedError(f"Response blocked by Gemini safety filters: finish_reason={finish_reason}")
    if finish_reason.upper() == "MAX_TOKENS":
        raise PartialResponseError(
            "Gemini response was cut off (hit max_output_tokens) before completing the JSON."
        )

    # --- Empty / missing text ---
    text = response.text
    if not text or not text.strip():
        raise EmptyResponseError("Gemini returned an empty response body")

    return text


def extractor(messy_text: str, max_retries: int = 3):
    """
    Extracts structured GST invoice data from unstructured text using Gemini.

    Two layers of retry are at play here, deliberately kept separate:
      1. `_call_gemini` (tenacity): retries the SAME prompt on transient
         network failures -- 429 rate limits, 5xx, timeouts, empty bodies.
      2. This loop: retries with a REVISED prompt when Gemini responds
         successfully but the content itself is wrong -- malformed JSON or
         a schema mismatch. Retrying the same prompt here would be pointless;
         Gemini needs the error fed back to it to self-correct.

    Args:
        messy_text: Raw invoice text (OCR, pasted email, etc.)
        max_retries: Number of self-correction attempts if validation fails (default 3)

    Returns:
        Details: A validated Pydantic model instance with extracted invoice data

    Raises:
        ValidationError: If all self-correction attempts fail to produce valid JSON.
        SafetyBlockedError: If Gemini blocks the prompt/response on safety grounds
            (never retried automatically -- the caller must alter the input).
        PartialResponseError: If output kept getting cut off by max_output_tokens
            after all self-correction attempts.
        RetryError: If network-level failures (429/5xx/timeout) persisted past
            tenacity's retry budget.
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=gst_basemodel.Details,
        # Belt-and-braces against runaway/truncated generations.
        max_output_tokens=4096,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    prompt = f"Extract the GST invoice details from this text:\n\n{messy_text}"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_json = _call_gemini(prompt, config)
        except SafetyBlockedError:
            # Retrying with the same (or an auto-mutated) prompt won't help;
            # surface this immediately rather than burning retries on it.
            logger.error("Extraction aborted: content was blocked by safety filters.")
            raise
        except RetryError as exc:
            # tenacity exhausted its backoff budget on a persistent network
            # failure (e.g. sustained 429s). Don't silently keep looping --
            # surface the underlying cause.
            logger.error("Gemini call failed after repeated backoff retries: %s", exc)
            raise
        except PartialResponseError as exc:
            # Output kept getting cut off. Worth one self-correction pass
            # asking for terser output, same as a schema-repair retry.
            last_error = exc
            logger.warning("[attempt %d/%d] response truncated: %s", attempt, max_retries, exc)
            prompt = (
                "Your previous response was cut off before it finished (hit the "
                "token limit). Re-extract the same data but be as concise as "
                "possible -- no extra commentary, no whitespace, minimal formatting. "
                f"Return ONLY the JSON.\n\nOriginal text:\n{messy_text}"
            )
            continue

        try:
            # model_validate_json() parses the JSON string AND validates it
            # against the Details schema in one step. Given the real schema
            # (amount: float, date: date), this already rejects non-numeric
            # amounts and invalid dates -- no separate business-rule layer needed.
            details = gst_basemodel.Details.model_validate_json(raw_json)
            return details

        except ValidationError as exc:
            last_error = exc
            logger.warning("[attempt %d/%d] schema validation failed:\n%s", attempt, max_retries, exc)
            prompt = (
                f"Your previous JSON output failed schema validation with this error:\n"
                f"{exc}\n\n"
                f"Previous output was:\n{raw_json}\n\n"
                f"Re-extract the GST invoice details from the ORIGINAL text below, "
                f"fixing the schema issue. Return ONLY valid JSON matching the schema.\n\n"
                f"Original text:\n{messy_text}"
            )

        except Exception as exc:
            # Gemini's response wasn't even valid JSON (rare, since
            # response_mime_type="application/json" is set, but not impossible).
            last_error = exc
            logger.warning(
                "[attempt %d/%d] could not parse response as JSON: %s", attempt, max_retries, exc
            )
            prompt = (
                f"Your previous response was not valid JSON. Return ONLY valid JSON "
                f"matching the required schema, with no markdown fences or commentary.\n\n"
                f"Original text:\n{messy_text}"
            )

    # All self-correction attempts exhausted - raise the last content-level error.
    raise last_error


if __name__ == "__main__":
    sample_messy_text = """
    TAX INVOICE
    M/s Sharma Traders & Co.
    GSTIN: 29ABCDE1234F1Z5
    Invoice Date: 14/03/2024

    Sl  Item                  Qty   Rate     Amt
    1   A4 Paper Ream         10    320.00   3200.00
    2   Stapler (Heavy Duty)  3     150.50   451.50
    3   Blue Ink Pens (Box)   5     90.00    450.00

    Sub total : 4101.50
    CGST 9%   : 369.14
    SGST 9%   : 369.14
    -----------------------------------------
    Grand Total: 4839.78
    Thank you for your business!
    """

    result = extractor(sample_messy_text)
    print(result.model_dump_json(indent=2))
