#!/usr/bin/env python3
"""
Sprint 4 Task 4: Correctness Verification for MMA VerdictMoE.

Tests the MMA kernel pipeline in full vLLM inference:
1. 4-prompt coherence test (greedy, temperature=0)
2. Perplexity comparison against CUTLASS baseline (logprob delta)
3. Extended 2000-token generation test (repetition, garbage, coherence)

Usage:
    # Against MMA backend (VLLM_USE_VERDICT_MOE=1, VLLM_VERDICT_MMA=1):
    python3 test_correctness_mma.py --url http://localhost:9200

    # Full comparison (run CUTLASS server on port 9201, MMA on 9200):
    python3 test_correctness_mma.py --url http://localhost:9200 --baseline-url http://localhost:9201

    # Just coherence test:
    python3 test_correctness_mma.py --url http://localhost:9200 --test coherence

    # Just perplexity comparison:
    python3 test_correctness_mma.py --url http://localhost:9200 --baseline-url http://localhost:9201 --test perplexity

    # Just extended generation:
    python3 test_correctness_mma.py --url http://localhost:9200 --test extended
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================
MODEL = "qwen3.5-397b-nvfp4"

COHERENCE_PROMPTS = [
    {
        "name": "factual_geography",
        "prompt": "The capital of Kentucky is",
        "max_tokens": 50,
        "check_keywords": ["Frankfort", "Kentucky"],
        "description": "Should name Frankfort as the capital",
    },
    {
        "name": "code_generation",
        "prompt": 'def fibonacci(n):\n    """Return the nth Fibonacci number."""',
        "max_tokens": 100,
        "check_keywords": ["return", "fibonacci", "if", "else"],
        "description": "Should produce valid Python with recursion or iteration",
    },
    {
        "name": "science_explanation",
        "prompt": "Explain quantum entanglement in one sentence:",
        "max_tokens": 60,
        "check_keywords": ["particle", "quantum", "state", "measur"],
        "description": "Should mention particles, states, or measurement",
    },
    {
        "name": "long_essay",
        "prompt": "Write a detailed essay about the history of artificial intelligence:",
        "max_tokens": 500,
        "check_keywords": ["Turing", "machine learning", "neural", "computer"],
        "description": "Should be a coherent multi-paragraph essay",
    },
]

PERPLEXITY_PROMPT = (
    "The development of artificial intelligence has been one of the most "
    "transformative technological achievements of the 21st century. From "
    "early rule-based systems to modern deep learning architectures, the "
    "field has evolved rapidly. Key milestones include the development of "
    "neural networks, the invention of backpropagation, and the rise of "
    "transformer-based language models."
)

EXTENDED_PROMPT = (
    "Write a comprehensive and detailed analysis of the evolution of "
    "computer architecture from the 1940s to the present day, covering "
    "vacuum tubes, transistors, integrated circuits, microprocessors, "
    "multi-core designs, GPUs, and modern AI accelerators. Include "
    "specific examples of influential machines and their impact on "
    "computing capabilities:"
)


# ============================================================================
# API helpers
# ============================================================================
def check_server(base_url: str) -> bool:
    """Verify vLLM server is running and responsive."""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"  Server OK at {base_url}. Models: {models}")
            return True
    except Exception as e:
        print(f"  Server NOT ready at {base_url}: {e}")
        return False


def generate_completion(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    logprobs: bool = False,
    top_logprobs: int = 0,
) -> dict:
    """
    Send a completion request (non-chat) with optional logprobs.
    Returns: {"text": str, "tokens": list, "logprobs": list|None, "finish_reason": str}
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = max(top_logprobs, 1)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  HTTP {e.code}: {body[:500]}")
        raise

    choice = result["choices"][0]
    text = choice.get("text", "")
    finish = choice.get("finish_reason", "unknown")

    lp_data = choice.get("logprobs")
    token_logprobs = None
    tokens = None
    if lp_data:
        token_logprobs = lp_data.get("token_logprobs", [])
        tokens = lp_data.get("tokens", [])

    return {
        "text": text,
        "tokens": tokens,
        "logprobs": token_logprobs,
        "finish_reason": finish,
    }


def generate_chat(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Chat completion (fallback if /v1/completions not available)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    choice = result["choices"][0]
    text = choice.get("message", {}).get("content", "")
    finish = choice.get("finish_reason", "unknown")
    return {"text": text, "finish_reason": finish}


def warmup_server(base_url: str, n: int = 2):
    """Warmup with short generations to trigger CUDA graph capture."""
    print(f"  Warming up server ({n} requests)...")
    for i in range(n):
        try:
            generate_completion(base_url, "Hello", 10, temperature=0.0)
        except Exception:
            try:
                generate_chat(base_url, "Hello", 10, temperature=0.0)
            except Exception:
                pass
    print("  Warmup complete.")


# ============================================================================
# Test 1: 4-Prompt Coherence Test
# ============================================================================
@dataclass
class CoherenceResult:
    name: str
    prompt: str
    output: str
    max_tokens: int
    passed: bool
    issues: list = field(default_factory=list)
    keywords_found: list = field(default_factory=list)
    keywords_missing: list = field(default_factory=list)


def check_coherence(text: str) -> list:
    """Check for signs of incoherent generation."""
    issues = []

    # Check for garbage / high non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127 and not c.isalpha())
    if len(text) > 0 and non_ascii / len(text) > 0.15:
        issues.append(f"High non-ASCII ratio: {non_ascii}/{len(text)}")

    # Check for excessive repetition (same 5+ word phrase repeated 3+ times)
    words = text.split()
    if len(words) > 20:
        for ngram_len in [5, 8, 12]:
            ngrams = [
                " ".join(words[i : i + ngram_len])
                for i in range(len(words) - ngram_len + 1)
            ]
            counts = Counter(ngrams)
            for phrase, count in counts.most_common(3):
                if count >= 3:
                    issues.append(
                        f"Repeated {ngram_len}-gram '{phrase[:50]}...' {count}x"
                    )

    # Check for very short output (might indicate early stop / garbage)
    if len(text.strip()) < 10:
        issues.append(f"Very short output: {len(text.strip())} chars")

    # Check for all-caps gibberish
    if len(text) > 50:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if caps_ratio > 0.5:
            issues.append(f"Excessive caps ratio: {caps_ratio:.2f}")

    return issues


def run_coherence_test(base_url: str) -> list:
    """Run 4-prompt coherence test with greedy decoding."""
    print("\n" + "=" * 70)
    print("TEST 1: 4-PROMPT COHERENCE TEST (greedy, temperature=0)")
    print("=" * 70)

    results = []
    use_chat = False

    # Try completions API first, fall back to chat
    try:
        generate_completion(base_url, "test", 5, temperature=0.0)
    except Exception:
        print("  /v1/completions not available, using /v1/chat/completions")
        use_chat = True

    for i, spec in enumerate(COHERENCE_PROMPTS):
        print(f"\n--- Prompt {i+1}/4: {spec['name']} ---")
        print(f"  Prompt: {spec['prompt'][:80]}...")
        print(f"  Max tokens: {spec['max_tokens']}")

        t0 = time.perf_counter()
        try:
            if use_chat:
                resp = generate_chat(
                    base_url, spec["prompt"], spec["max_tokens"], temperature=0.0
                )
            else:
                resp = generate_completion(
                    base_url, spec["prompt"], spec["max_tokens"], temperature=0.0
                )
            text = resp["text"]
        except Exception as e:
            text = ""
            print(f"  ERROR: {e}")

        elapsed = time.perf_counter() - t0
        print(f"  Generated in {elapsed:.2f}s")

        # Check keywords
        text_lower = text.lower()
        found = [kw for kw in spec["check_keywords"] if kw.lower() in text_lower]
        missing = [kw for kw in spec["check_keywords"] if kw.lower() not in text_lower]

        # Check coherence
        issues = check_coherence(text)

        # For code prompt, check syntactic validity
        if spec["name"] == "code_generation":
            full_code = spec["prompt"] + text
            try:
                compile(full_code, "<test>", "exec")
                print("  Code syntax: VALID")
            except SyntaxError as e:
                # Incomplete code is OK if it's coherent
                if "unexpected EOF" not in str(e):
                    issues.append(f"Syntax error: {e}")
                    print(f"  Code syntax: ERROR ({e})")
                else:
                    print("  Code syntax: incomplete but valid structure")

        passed = len(issues) == 0 and len(found) >= 1
        status = "PASS" if passed else "FAIL"

        print(f"  Output ({len(text)} chars): {text[:200]}...")
        print(f"  Keywords found: {found}")
        if missing:
            print(f"  Keywords missing: {missing}")
        if issues:
            print(f"  Issues: {issues}")
        print(f"  Result: {status}")

        results.append(
            CoherenceResult(
                name=spec["name"],
                prompt=spec["prompt"],
                output=text,
                max_tokens=spec["max_tokens"],
                passed=passed,
                issues=issues,
                keywords_found=found,
                keywords_missing=missing,
            )
        )

    all_passed = all(r.passed for r in results)
    print(f"\n{'='*70}")
    print(f"COHERENCE TEST SUMMARY: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    for r in results:
        print(f"  {r.name}: {'PASS' if r.passed else 'FAIL'} "
              f"(keywords: {len(r.keywords_found)}/{len(r.keywords_found)+len(r.keywords_missing)}, "
              f"issues: {len(r.issues)})")
    print(f"{'='*70}")

    return results


# ============================================================================
# Test 2: Perplexity Comparison
# ============================================================================
@dataclass
class PerplexityResult:
    mma_logprobs: list
    baseline_logprobs: list
    avg_delta: float
    max_delta: float
    mma_perplexity: float
    baseline_perplexity: float
    ppl_ratio: float
    passed: bool
    details: str = ""


def compute_perplexity(logprobs: list) -> float:
    """Compute perplexity from per-token log probabilities."""
    valid = [lp for lp in logprobs if lp is not None and math.isfinite(lp)]
    if not valid:
        return float("inf")
    avg_nll = -sum(valid) / len(valid)
    return math.exp(avg_nll)


def run_perplexity_test(mma_url: str, baseline_url: str) -> PerplexityResult:
    """Compare per-token logprobs between MMA and CUTLASS baseline."""
    print("\n" + "=" * 70)
    print("TEST 2: PERPLEXITY COMPARISON (MMA vs CUTLASS)")
    print("=" * 70)

    max_tokens = 100

    print(f"\n  Prompt: {PERPLEXITY_PROMPT[:80]}...")
    print(f"  Generating {max_tokens} tokens from each backend...")

    # Generate from MMA backend with logprobs
    print("\n  [MMA backend]")
    t0 = time.perf_counter()
    mma_resp = generate_completion(
        mma_url, PERPLEXITY_PROMPT, max_tokens,
        temperature=0.0, logprobs=True, top_logprobs=1,
    )
    mma_time = time.perf_counter() - t0
    print(f"    Generated {len(mma_resp['text'])} chars in {mma_time:.2f}s")

    # Generate from CUTLASS baseline with logprobs
    print("  [CUTLASS baseline]")
    t0 = time.perf_counter()
    baseline_resp = generate_completion(
        baseline_url, PERPLEXITY_PROMPT, max_tokens,
        temperature=0.0, logprobs=True, top_logprobs=1,
    )
    baseline_time = time.perf_counter() - t0
    print(f"    Generated {len(baseline_resp['text'])} chars in {baseline_time:.2f}s")

    # Compare logprobs
    mma_lp = mma_resp["logprobs"] or []
    base_lp = baseline_resp["logprobs"] or []

    if not mma_lp or not base_lp:
        print("  WARNING: Logprobs not available from one or both servers!")
        print("  Falling back to text comparison only.")
        # Text similarity check
        mma_text = mma_resp["text"][:500]
        base_text = baseline_resp["text"][:500]
        # Simple word overlap
        mma_words = set(mma_text.lower().split())
        base_words = set(base_text.lower().split())
        overlap = len(mma_words & base_words) / max(len(mma_words | base_words), 1)
        print(f"  Text word overlap: {overlap:.2%}")
        return PerplexityResult(
            mma_logprobs=mma_lp,
            baseline_logprobs=base_lp,
            avg_delta=0.0,
            max_delta=0.0,
            mma_perplexity=0.0,
            baseline_perplexity=0.0,
            ppl_ratio=1.0,
            passed=overlap > 0.5,
            details=f"Logprobs unavailable. Text word overlap: {overlap:.2%}",
        )

    # Compute per-token logprob deltas
    min_len = min(len(mma_lp), len(base_lp))
    deltas = []
    for i in range(min_len):
        if mma_lp[i] is not None and base_lp[i] is not None:
            deltas.append(abs(mma_lp[i] - base_lp[i]))

    avg_delta = sum(deltas) / len(deltas) if deltas else float("inf")
    max_delta = max(deltas) if deltas else float("inf")

    mma_ppl = compute_perplexity(mma_lp)
    base_ppl = compute_perplexity(base_lp)
    ppl_ratio = mma_ppl / base_ppl if base_ppl > 0 else float("inf")

    # Thresholds
    avg_ok = avg_delta < 0.01
    ppl_ok = ppl_ratio < 1.01
    passed = avg_ok and ppl_ok

    print(f"\n  Results ({min_len} tokens compared):")
    print(f"    Avg logprob delta:  {avg_delta:.6f}  (threshold: < 0.01) {'PASS' if avg_ok else 'FAIL'}")
    print(f"    Max logprob delta:  {max_delta:.6f}")
    print(f"    MMA perplexity:     {mma_ppl:.4f}")
    print(f"    Baseline perplexity:{base_ppl:.4f}")
    print(f"    PPL ratio:          {ppl_ratio:.6f}  (threshold: < 1.01) {'PASS' if ppl_ok else 'FAIL'}")
    print(f"    Overall: {'PASS' if passed else 'FAIL'}")

    # Show first 10 token-level comparisons
    mma_tokens = mma_resp.get("tokens", [])
    base_tokens = baseline_resp.get("tokens", [])
    print(f"\n  Token-level comparison (first 10):")
    print(f"    {'#':>3} {'MMA token':>20} {'MMA lp':>10} {'Base token':>20} {'Base lp':>10} {'Delta':>10}")
    for i in range(min(10, min_len)):
        mt = mma_tokens[i] if i < len(mma_tokens) else "?"
        bt = base_tokens[i] if i < len(base_tokens) else "?"
        ml = mma_lp[i] if mma_lp[i] is not None else 0
        bl = base_lp[i] if base_lp[i] is not None else 0
        d = abs(ml - bl)
        print(f"    {i:>3} {repr(mt):>20} {ml:>10.4f} {repr(bt):>20} {bl:>10.4f} {d:>10.6f}")

    details = (
        f"Compared {min_len} tokens. "
        f"Avg delta={avg_delta:.6f}, Max delta={max_delta:.6f}, "
        f"PPL ratio={ppl_ratio:.6f}"
    )

    return PerplexityResult(
        mma_logprobs=mma_lp[:min_len],
        baseline_logprobs=base_lp[:min_len],
        avg_delta=avg_delta,
        max_delta=max_delta,
        mma_perplexity=mma_ppl,
        baseline_perplexity=base_ppl,
        ppl_ratio=ppl_ratio,
        passed=passed,
        details=details,
    )


# ============================================================================
# Test 3: Extended Generation Test (2000 tokens)
# ============================================================================
@dataclass
class ExtendedResult:
    text: str
    token_count: int
    has_repetition_loops: bool
    has_garbage_tokens: bool
    coherence_degradation: bool
    passed: bool
    issues: list = field(default_factory=list)


def detect_repetition_loops(text: str, min_phrase_len: int = 20) -> list:
    """Detect repetition loops in generated text."""
    issues = []
    words = text.split()

    # Check for repeated sentences (approx)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if sentences:
        sent_counts = Counter(sentences)
        for sent, count in sent_counts.most_common(5):
            if count >= 3:
                issues.append(
                    f"Sentence repeated {count}x: '{sent[:60]}...'"
                )

    # Check for repeated n-grams (5, 10, 20 words)
    for n in [5, 10, 20]:
        if len(words) < n * 3:
            continue
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        for phrase, count in counts.most_common(3):
            threshold = max(3, len(words) // (n * 10))
            if count >= threshold:
                issues.append(
                    f"{n}-gram repeated {count}x (threshold {threshold}): "
                    f"'{phrase[:60]}...'"
                )

    # Check for character-level repetition (same char 20+ times)
    for match in re.finditer(r'(.)\1{19,}', text):
        issues.append(f"Character '{match.group(1)}' repeated {len(match.group())}x")

    return issues


def detect_garbage(text: str) -> list:
    """Detect garbage tokens / encoding artifacts."""
    issues = []

    # High ratio of special/control characters
    control = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if control > 10:
        issues.append(f"Control characters: {control}")

    # Excessive non-printable Unicode
    non_print = sum(1 for c in text if not c.isprintable() and c not in '\n\r\t')
    if len(text) > 100 and non_print / len(text) > 0.05:
        issues.append(f"Non-printable ratio: {non_print}/{len(text)} = {non_print/len(text):.2%}")

    # Random byte sequences (consecutive non-ASCII non-letter chars)
    for match in re.finditer(r'[^\x00-\x7f]{10,}', text):
        # Check if it's legitimate non-English text or garbage
        segment = match.group()
        if not any(c.isalpha() for c in segment):
            issues.append(f"Non-ASCII garbage at pos {match.start()}: {repr(segment[:40])}")

    return issues


def check_coherence_degradation(text: str) -> list:
    """Check if text quality degrades significantly in the second half."""
    issues = []
    if len(text) < 200:
        return issues

    mid = len(text) // 2
    first_half = text[:mid]
    second_half = text[mid:]

    # Compare word diversity
    first_words = first_half.split()
    second_words = second_half.split()

    if len(first_words) > 20 and len(second_words) > 20:
        first_diversity = len(set(first_words)) / len(first_words)
        second_diversity = len(set(second_words)) / len(second_words)

        if second_diversity < first_diversity * 0.5:
            issues.append(
                f"Vocabulary diversity dropped: {first_diversity:.2f} → {second_diversity:.2f} "
                f"(second half has {second_diversity/first_diversity:.0%} of first half diversity)"
            )

    # Check if second half has more repetition
    second_rep = detect_repetition_loops(second_half)
    if len(second_rep) > 0:
        issues.append(f"Second half has repetition issues: {second_rep[0]}")

    return issues


def run_extended_test(base_url: str) -> ExtendedResult:
    """Generate 2000 tokens and check for quality degradation."""
    print("\n" + "=" * 70)
    print("TEST 3: EXTENDED GENERATION TEST (2000 tokens)")
    print("=" * 70)

    max_tokens = 2000
    print(f"\n  Prompt: {EXTENDED_PROMPT[:80]}...")
    print(f"  Generating {max_tokens} tokens...")

    t0 = time.perf_counter()
    use_chat = False
    try:
        resp = generate_completion(
            base_url, EXTENDED_PROMPT, max_tokens, temperature=0.0
        )
        text = resp["text"]
    except Exception:
        use_chat = True
        resp = generate_chat(base_url, EXTENDED_PROMPT, max_tokens, temperature=0.0)
        text = resp["text"]

    elapsed = time.perf_counter() - t0
    word_count = len(text.split())
    print(f"  Generated {len(text)} chars, ~{word_count} words in {elapsed:.1f}s")
    print(f"  Finish reason: {resp.get('finish_reason', 'unknown')}")

    # Run checks
    print("\n  Running quality checks...")

    rep_issues = detect_repetition_loops(text)
    garbage_issues = detect_garbage(text)
    degradation_issues = check_coherence_degradation(text)
    general_issues = check_coherence(text)

    all_issues = []
    has_rep = len(rep_issues) > 0
    has_garbage = len(garbage_issues) > 0
    has_degradation = len(degradation_issues) > 0

    if rep_issues:
        all_issues.extend([f"[REPETITION] {i}" for i in rep_issues])
    if garbage_issues:
        all_issues.extend([f"[GARBAGE] {i}" for i in garbage_issues])
    if degradation_issues:
        all_issues.extend([f"[DEGRADATION] {i}" for i in degradation_issues])
    if general_issues:
        all_issues.extend([f"[COHERENCE] {i}" for i in general_issues])

    passed = not has_rep and not has_garbage and not has_degradation and not general_issues

    print(f"\n  Repetition loops:      {'DETECTED' if has_rep else 'NONE'}")
    print(f"  Garbage tokens:        {'DETECTED' if has_garbage else 'NONE'}")
    print(f"  Coherence degradation: {'DETECTED' if has_degradation else 'NONE'}")
    if all_issues:
        print(f"\n  Issues found:")
        for issue in all_issues:
            print(f"    - {issue}")

    print(f"\n  First 300 chars:")
    print(f"    {text[:300]}")
    print(f"\n  Last 300 chars:")
    print(f"    {text[-300:]}")

    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return ExtendedResult(
        text=text,
        token_count=max_tokens,
        has_repetition_loops=has_rep,
        has_garbage_tokens=has_garbage,
        coherence_degradation=has_degradation,
        passed=passed,
        issues=all_issues,
    )


# ============================================================================
# Report generation
# ============================================================================
def generate_report(
    coherence_results: list | None,
    ppl_result: PerplexityResult | None,
    extended_result: ExtendedResult | None,
    output_path: str,
):
    """Generate markdown report."""
    lines = []
    lines.append("# Sprint 4 Task 4: Correctness Verification\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Overall status
    all_pass = True
    if coherence_results and not all(r.passed for r in coherence_results):
        all_pass = False
    if ppl_result and not ppl_result.passed:
        all_pass = False
    if extended_result and not extended_result.passed:
        all_pass = False

    lines.append(f"**Status**: {'PASS — All correctness tests passed' if all_pass else 'FAIL — See details below'}\n")
    lines.append("---\n")

    # Test 1: Coherence
    if coherence_results:
        lines.append("## 1. 4-Prompt Coherence Test (greedy, temperature=0)\n")
        coh_pass = all(r.passed for r in coherence_results)
        lines.append(f"**Result: {'PASS' if coh_pass else 'FAIL'}**\n")
        lines.append("| # | Prompt | Keywords | Issues | Status |")
        lines.append("|---|--------|----------|--------|--------|")
        for i, r in enumerate(coherence_results):
            kw = f"{len(r.keywords_found)}/{len(r.keywords_found)+len(r.keywords_missing)}"
            iss = str(len(r.issues))
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"| {i+1} | {r.name} | {kw} | {iss} | {status} |")

        for i, r in enumerate(coherence_results):
            lines.append(f"\n### Prompt {i+1}: {r.name}\n")
            lines.append(f"**Prompt**: `{r.prompt[:100]}`\n")
            lines.append(f"**Output** ({len(r.output)} chars):\n")
            lines.append(f"```\n{r.output[:1000]}\n```\n")
            if r.issues:
                lines.append(f"**Issues**: {', '.join(r.issues)}\n")
            if r.keywords_missing:
                lines.append(f"**Missing keywords**: {', '.join(r.keywords_missing)}\n")

        lines.append("---\n")

    # Test 2: Perplexity
    if ppl_result:
        lines.append("## 2. Perplexity Comparison (MMA vs CUTLASS)\n")
        lines.append(f"**Result: {'PASS' if ppl_result.passed else 'FAIL'}**\n")
        lines.append("| Metric | Value | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")
        avg_ok = ppl_result.avg_delta < 0.01
        ppl_ok = ppl_result.ppl_ratio < 1.01
        lines.append(f"| Avg logprob delta | {ppl_result.avg_delta:.6f} | < 0.01 | {'PASS' if avg_ok else 'FAIL'} |")
        lines.append(f"| Max logprob delta | {ppl_result.max_delta:.6f} | — | — |")
        lines.append(f"| MMA perplexity | {ppl_result.mma_perplexity:.4f} | — | — |")
        lines.append(f"| Baseline perplexity | {ppl_result.baseline_perplexity:.4f} | — | — |")
        lines.append(f"| PPL ratio (MMA/base) | {ppl_result.ppl_ratio:.6f} | < 1.01 | {'PASS' if ppl_ok else 'FAIL'} |")
        if ppl_result.details:
            lines.append(f"\n{ppl_result.details}\n")
        lines.append("\n---\n")

    # Test 3: Extended generation
    if extended_result:
        lines.append("## 3. Extended Generation Test (2000 tokens)\n")
        lines.append(f"**Result: {'PASS' if extended_result.passed else 'FAIL'}**\n")
        lines.append("| Check | Result |")
        lines.append("|-------|--------|")
        lines.append(f"| Repetition loops | {'NONE' if not extended_result.has_repetition_loops else 'DETECTED'} |")
        lines.append(f"| Garbage tokens | {'NONE' if not extended_result.has_garbage_tokens else 'DETECTED'} |")
        lines.append(f"| Coherence degradation | {'NONE' if not extended_result.coherence_degradation else 'DETECTED'} |")

        if extended_result.issues:
            lines.append("\n**Issues found:**\n")
            for issue in extended_result.issues:
                lines.append(f"- {issue}")
            lines.append("")

        lines.append(f"\n**Full output** ({len(extended_result.text)} chars):\n")
        lines.append(f"```\n{extended_result.text[:3000]}\n```\n")
        if len(extended_result.text) > 3000:
            lines.append(f"\n... ({len(extended_result.text) - 3000} chars truncated)\n")
            lines.append(f"\n**Last 500 chars:**\n```\n{extended_result.text[-500:]}\n```\n")

        lines.append("---\n")

    # Final summary
    lines.append("## Summary\n")
    lines.append("| Test | Status |")
    lines.append("|------|--------|")
    if coherence_results:
        coh_pass = all(r.passed for r in coherence_results)
        lines.append(f"| 4-Prompt Coherence | {'PASS' if coh_pass else 'FAIL'} |")
    if ppl_result:
        lines.append(f"| Perplexity Comparison | {'PASS' if ppl_result.passed else 'FAIL'} |")
    if extended_result:
        lines.append(f"| Extended Generation (2000 tok) | {'PASS' if extended_result.passed else 'FAIL'} |")
    lines.append(f"| **Overall** | **{'PASS' if all_pass else 'FAIL'}** |")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")

    return report


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MMA VerdictMoE Correctness Verification")
    parser.add_argument("--url", default="http://localhost:9200",
                        help="vLLM server URL (MMA backend)")
    parser.add_argument("--baseline-url", default=None,
                        help="vLLM server URL (CUTLASS baseline) for perplexity comparison")
    parser.add_argument("--test", default="all",
                        choices=["all", "coherence", "perplexity", "extended"],
                        help="Which test to run")
    parser.add_argument("--output", default=None,
                        help="Output report path (default: ~/verdict_sprint4_logs/task4_correctness.md)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    output_path = args.output or str(
        Path.home() / "verdict_sprint4_logs" / "task4_correctness.md"
    )

    print("=" * 70)
    print("SPRINT 4 TASK 4: MMA VerdictMoE Correctness Verification")
    print("=" * 70)
    print(f"  MMA server:      {args.url}")
    print(f"  Baseline server:  {args.baseline_url or 'N/A (skipping perplexity test)'}")
    print(f"  Test:             {args.test}")
    print(f"  Model:            {MODEL}")
    print(f"  Output:           {output_path}")

    # Check servers
    print("\nChecking servers...")
    if not check_server(args.url):
        print("\nERROR: MMA server not available. Start vLLM with:")
        print("  VLLM_USE_VERDICT_MOE=1 VLLM_VERDICT_MMA=1 python -m vllm.entrypoints.openai.api_server ...")
        sys.exit(1)

    if args.baseline_url and args.test in ["all", "perplexity"]:
        if not check_server(args.baseline_url):
            print("\nWARNING: Baseline server not available. Skipping perplexity test.")
            if args.test == "perplexity":
                sys.exit(1)
            args.baseline_url = None

    # Warmup
    if not args.no_warmup:
        warmup_server(args.url)
        if args.baseline_url:
            warmup_server(args.baseline_url)

    # Run tests
    coherence_results = None
    ppl_result = None
    extended_result = None

    if args.test in ["all", "coherence"]:
        coherence_results = run_coherence_test(args.url)

    if args.test in ["all", "perplexity"] and args.baseline_url:
        ppl_result = run_perplexity_test(args.url, args.baseline_url)

    if args.test in ["all", "extended"]:
        extended_result = run_extended_test(args.url)

    # Generate report
    report = generate_report(coherence_results, ppl_result, extended_result, output_path)

    # Final verdict
    print("\n" + "=" * 70)
    all_pass = True
    if coherence_results and not all(r.passed for r in coherence_results):
        all_pass = False
    if ppl_result and not ppl_result.passed:
        all_pass = False
    if extended_result and not extended_result.passed:
        all_pass = False

    if all_pass:
        print("OVERALL VERDICT: PASS")
        print("All correctness tests passed. Safe to proceed to benchmark.")
    else:
        print("OVERALL VERDICT: FAIL")
        print("Correctness issues detected. Fix before proceeding.")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
