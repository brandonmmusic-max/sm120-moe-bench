"""
KLC Real-World Legal Benchmark

Tests decode throughput under realistic conditions with diverse legal prompts
that exercise different MoE expert routing patterns:
- Short factual questions (few experts, fast routing)
- Citation normalization (specialized legal knowledge)
- Trick/hallucination questions (model must reason carefully)
- Complex multi-factor analysis (deep expert engagement)
- Messy real-world inputs with irrelevant details (noise filtering)

Matrix: 1/2/4 users × 1K/2K/3K/4K output tokens × 24 diverse prompts
"""

import asyncio
import json
import time
import sys
from dataclasses import dataclass
from statistics import mean, median

import httpx

API_URL = "http://localhost:9200/v1/chat/completions"
MODEL = "qwen3.5-397b-nvfp4"

# === PROMPT CATEGORIES ===
# Drawn from KLC benchmarks: prbench_ky, gdpval, v16

PROMPTS = {
    # --- SHORT FACTUAL (quick expert routing, short answers expected) ---
    "short_statute": "What is the statute of limitations for personal injury claims in Kentucky?",

    "short_dui": "What is the legal blood alcohol limit for DUI in Kentucky?",

    "short_contract": "What is the statute of limitations for breach of a written contract in Kentucky?",

    # --- CITATION NORMALIZATION (specialized legal knowledge) ---
    "cite_comparative": "Explain the difference between KRS 411.182 comparative fault and the old contributory negligence bar rule. When did Kentucky change and what cases led to the change?",

    "cite_broad_form": "How does Kentucky law address the broad form deed for mineral rights? Cite the relevant constitutional amendment, KRS sections, and key case law including Ward v. Harding.",

    "cite_summary_judgment": "What standard does Kentucky apply for summary judgment motions? How does the Steelvest standard differ from the federal Celotex standard? Cite the relevant civil rules and case law.",

    # --- TRICK / HALLUCINATION (model must not fabricate) ---
    "trick_holographic": "Are holographic wills valid in Kentucky? What about a holographic will executed in Tennessee by a Kentucky domiciliary?",

    "trick_castle": "Under what circumstances does Kentucky's castle doctrine authorize the use of deadly force? Is there a duty to retreat in your own home?",

    "trick_grandparent": "Do grandparents have an absolute right to visitation in Kentucky? Can a grandparent petition for custody?",

    # --- COMPLEX MULTI-FACTOR ANALYSIS (deep reasoning, many experts) ---
    "complex_divorce": "Representing wife in a contested divorce. Married 18 years, 2 kids (14 and 11). Husband owns a plumbing business he started 3 years before marriage but grew substantially during. He's claiming the whole business is non-marital. Wife was a SAHM for 12 years, just started working part-time making $28k. Husband's pulling about $180k between salary and distributions. He also just moved his girlfriend into the marital home while wife and kids are at her mom's. We filed for a DVO but it was denied - no physical violence, just emotional stuff. Judge set temporary support at $2,800/mo which barely covers rent. What's my best strategy for the property hearing, and should I push for maintenance or focus on getting a bigger property split?",

    "complex_mineral": "Got a client who bought 40 acres in Carter County 5 years ago. Previous owner had severed mineral rights back in 1962 with a broad form deed. Now an energy company shows up wanting to do horizontal drilling and they're citing the old mineral deed. Client says no way - the surface damage would destroy his cattle operation. I know about the '88 constitutional amendment but the mineral deed has some unusual language granting 'all methods necessary.' What's my argument to block this and do I have any leverage?",

    "complex_pi": "Rear-end collision on I-75 in Madison County. My client (plaintiff) was stopped in traffic, got hit from behind by a semi doing about 40. Clear liability. Client has a herniated disc at L4-L5, had a microdiscectomy 4 months after the wreck. Was doing great, then reinjured it lifting something at work 8 months later. Defense is now saying the surgery failed because of the workplace reinjury, not the original wreck. Client's total meds are about $180k. Lost wages around $45k so far, and the surgeon says he'll need a fusion eventually - another $120k. The trucker's company has a $1M policy. What's my damages theory and how do I deal with the reinjury problem?",

    # --- MESSY REAL-WORLD (noise, irrelevant details, misspellings) ---
    "messy_eviction": "Client owns a duplex in Lexington, tenant in unit B hasn't paid rent in 4 months ($1,200/mo so $4,800 owed). Lease expired 6 months ago but tenant stayed and client kept accepting rent - until the checks stopped coming. Now tenant says she's withholding rent because the furnace hasn't worked right all winter and there's mold in the bathroom. Client admits the furnace is old but says it works, just not great. Haven't looked at the mold. Tenant also has 2 unauthorized dogs in violation of the lease. Client wants her out yesterday. What's the fastest legal path and am I exposed on the habitability stuff?",

    "messy_dui_suppress": "Defending a DUI case in Hardin County. Client blew a .12 on the breathalyzer at the station. Here's the wrinkle - the officer used an Intoxilyzer 9000 that had been certified 13 months ago, and the regulation says recertification every 12 months. Also, the officer's own certification to operate the machine expired 2 months before the arrest. Client's a commercial truck driver and a conviction means he loses his CDL permanently for a first offense because commercial limit is .04. Judge is known to be tough on DUI. What's my play?",

    "messy_slip_fall": "I was attacked in an apartment complex parking lot at night. There were no security guards and the lights were broken. The owner says crime wasn't foreseeable. Also, my neighbor is annoying but I don't think that's related. The police report shows 3 muggings in the same lot in the past year. I have medical bills of $23,000 and missed 6 weeks of work.",

    # --- EMPLOYMENT & DISCRIMINATION ---
    "employment_age": "Client is a restaurant manager in Louisville, 54 years old, been there 15 years. New regional VP comes in, starts making comments about 'fresh ideas' and 'young energy,' reassigns my client's best shifts to younger managers, puts her on a PIP with impossible metrics nobody else has to meet, and terminates her 3 months later for 'performance.' She's never had a bad review before the new VP. Oh, and the new VP hired his 28-year-old friend to replace her. She wants to know if she has a case and how much it's worth.",

    # --- WILL CONTEST / ESTATE ---
    "estate_contest": "Dad died last month, left a will from 2019 leaving everything to his second wife. Three adult kids from first marriage are furious. The will was drafted by a local attorney and properly witnessed. But here's the thing - dad was diagnosed with moderate dementia in 2018, and the kids have medical records showing cognitive decline. Second wife drove dad to the attorney's office and was in the waiting room. Kids want to contest. Estate is about $800k - house, farm equipment, and a $300k life insurance policy naming second wife as beneficiary. What are the kids' realistic chances on a will contest, and is there any play on the life insurance?",

    # --- CONSTRUCTION CONTRACT ---
    "construction_lien": "Small construction company client entered a fixed-price contract to build a 4,000 sq ft house in Fayette County for $450k. Midway through, homeowner started demanding upgrades - granite counters, upgraded HVAC, custom tile work. My client did the extras based on verbal promises to pay, never got change orders signed. Now the house is done, homeowner refuses to pay the extra $85k, says 'that was included in the original price.' My client hasn't been paid the final draw of $112k either. The house is worth about $620k. What are my options and how fast can I get my client paid?",

    # --- CUSTODY MODIFICATION ---
    "custody_mod": "Custody modification situation. Original decree from 3 years ago gave mom primary custody, dad gets every other weekend. Dad's remarried, stable home, making good money as an electrician. Mom just got her third DUI - this time with the kids in the car, ages 6 and 9. Kids were placed with dad temporarily by CPS. Mom did 30 days in jail, just got out, completed a 28-day inpatient program, and now wants the kids back. Dad filed for custody modification. Mom's attorney is arguing the treatment shows she's rehabilitated and there's no changed circumstances beyond a 'single incident.' What's my argument for dad?",

    # --- MEDICAL MALPRACTICE ---
    "medmal_procedure": "Client walked into a med mal case. 62-year-old guy went in for a routine knee replacement at a hospital in Kenton County. Surgeon nicked the popliteal artery during the procedure, didn't catch it in the OR, and by the time they figured it out 6 hours later in recovery, client lost so much blood he had a stroke. Now he's got permanent left-side weakness, can't drive, needs daily assistance. He was a self-employed home inspector making about $95k/year, planned to work until 70. The surgical note says 'unremarkable procedure' - no mention of the arterial damage. What do I need to do before I can even file this?",

    # --- CRIMINAL DEFENSE (4th Amendment) ---
    "criminal_search": "Client got pulled over on I-64 in Rowan County for 'weaving.' Officer says he smelled marijuana, searched the car, found 2 oz of weed and a handgun. Client has a valid CCDW permit for the gun. Problem is - dashcam shows the car barely touched the lane line once. No field sobriety test was offered. Officer went straight to 'I smell marijuana' and searched. Client has no priors, works as a nurse, and a conviction would cost her license. What's my suppression argument and what am I looking at if we lose the motion?",

    # --- WORKERS COMP / BLACK LUNG ---
    "workers_comp": "Workers' comp case in Perry County. Coal miner with 22 years underground, diagnosed with coal workers' pneumoconiosis (black lung) - both simple and complicated. He's 58, can't work underground anymore. His last employer went bankrupt 2 years ago and their workers comp carrier says coverage ended when the company folded. Previous employer (15 years there) says it's not their problem because the last employer should be responsible. The miner is getting sicker and needs benefits now. Where do I file and who pays?",

    # --- PARTITION / FAMILY FARM ---
    "partition_farm": "Two siblings inherited their parents' 120-acre farm in Shelby County equally. Brother wants to sell to a developer for $1.8M. Sister refuses - she lives on the property, has maintained it for 15 years, and runs a small hay operation. They can't agree on anything. Brother wants to force a sale. What are sister's options to keep the property or at least get fair value?",
}

@dataclass
class Result:
    prompt_key: str
    category: str
    tokens: int
    time: float
    ttft: float
    tps: float
    error: str = ""


def get_category(key):
    if key.startswith("short_"): return "short_factual"
    if key.startswith("cite_"): return "citation"
    if key.startswith("trick_"): return "trick"
    if key.startswith("complex_"): return "complex_analysis"
    if key.startswith("messy_"): return "messy_real_world"
    return "specialized"


async def stream_request(client, prompt_key, prompt, max_tokens):
    """Send a streaming chat completion and measure TTFT + decode speed."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a Kentucky legal expert. Provide thorough, well-cited analysis under Kentucky law. Cite specific KRS sections and case law where applicable."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.monotonic()
    t_first = None
    token_count = 0
    usage_tokens = None

    try:
        async with client.stream("POST", API_URL, json=payload,
                                  timeout=httpx.Timeout(600.0, connect=30.0)) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if usage and "completion_tokens" in usage:
                    usage_tokens = usage["completion_tokens"]

                if "choices" not in data or len(data["choices"]) == 0:
                    continue
                delta = data["choices"][0].get("delta", {})
                text = (delta.get("reasoning") or "") + (delta.get("content") or "")
                if text:
                    if t_first is None:
                        t_first = time.monotonic()
                    token_count += 1

    except Exception as e:
        elapsed = time.monotonic() - t_start
        return Result(prompt_key, get_category(prompt_key), 0, elapsed, 0, 0, str(e))

    elapsed = time.monotonic() - t_start
    final_tokens = usage_tokens if usage_tokens else token_count
    ttft = (t_first - t_start) if t_first else elapsed
    tps = final_tokens / elapsed if elapsed > 0 else 0

    return Result(prompt_key, get_category(prompt_key), final_tokens, elapsed, ttft, tps)


async def run_batch(client, prompt_items, max_tokens, concurrency):
    """Run a batch of prompts at given concurrency, return results."""
    # Select prompts round-robin up to concurrency
    selected = []
    keys = list(prompt_items.keys())
    for i in range(concurrency):
        k = keys[i % len(keys)]
        selected.append((k, prompt_items[k]))

    tasks = [stream_request(client, k, p, max_tokens) for k, p in selected]
    return await asyncio.gather(*tasks)


async def main():
    print("=" * 80)
    print("KLC REAL-WORLD LEGAL BENCHMARK")
    print(f"Model: {MODEL}")
    print(f"Prompts: {len(PROMPTS)} diverse legal scenarios")
    print(f"Concurrency: 1, 2, 4 users")
    print(f"Output tokens: 1K, 2K, 3K, 4K")
    print("=" * 80)

    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        # Warmup
        print("\nWarmup...")
        await stream_request(client, "warmup", "Hello, what is Kentucky?", 64)
        print("Warmup done.\n")

        all_results = []

        for max_tokens in [1024, 2048, 3072, 4096]:
            print(f"\n{'='*80}")
            print(f"OUTPUT TOKENS: {max_tokens}")
            print(f"{'='*80}")

            for concurrency in [1, 2, 4]:
                print(f"\n--- {concurrency} user(s), {max_tokens} max tokens ---")

                # For single user, cycle through ALL prompts
                # For multi-user, run 3 batches with different prompt sets
                if concurrency == 1:
                    # Run each prompt category once
                    category_keys = {}
                    for k, p in PROMPTS.items():
                        cat = get_category(k)
                        if cat not in category_keys:
                            category_keys[cat] = (k, p)

                    batch_results = []
                    for k, p in category_keys.values():
                        results = await run_batch(client, {k: p}, max_tokens, 1)
                        batch_results.extend(results)
                        r = results[0]
                        print(f"  {r.prompt_key:25s} {r.tokens:5d} tok  {r.time:6.1f}s  {r.tps:6.1f} tok/s  TTFT={r.ttft:.2f}s  [{r.category}]")

                    valid = [r for r in batch_results if not r.error]
                    if valid:
                        avg_tps = mean(r.tps for r in valid)
                        med_ttft = median(r.ttft for r in valid)
                        total_tok = sum(r.tokens for r in valid)
                        print(f"  >>> AVG: {avg_tps:.1f} tok/s  TTFT median: {med_ttft:.2f}s  total: {total_tok} tokens")

                else:
                    # Multi-user: run 3 batches with mixed prompts
                    prompt_list = list(PROMPTS.items())
                    batch_results = []
                    for batch_idx in range(3):
                        start = batch_idx * concurrency
                        batch_prompts = dict(prompt_list[start:start + concurrency])
                        if len(batch_prompts) < concurrency:
                            batch_prompts = dict(prompt_list[:concurrency])

                        t_start = time.monotonic()
                        results = await run_batch(client, batch_prompts, max_tokens, concurrency)
                        wall = time.monotonic() - t_start
                        batch_results.extend(results)

                        valid = [r for r in results if not r.error]
                        total_tok = sum(r.tokens for r in valid)
                        sys_tps = total_tok / wall if wall > 0 else 0
                        per_user = sys_tps / concurrency
                        print(f"  Batch {batch_idx+1}: sys={sys_tps:.1f} tok/s  per-user={per_user:.1f}  wall={wall:.1f}s  tokens={total_tok}")

                    valid = [r for r in batch_results if not r.error]
                    if valid:
                        all_tps = [r.tps for r in valid]
                        all_ttft = [r.ttft for r in valid]
                        print(f"  >>> AVG per-stream: {mean(all_tps):.1f} tok/s  TTFT median: {median(all_ttft):.2f}s")

                all_results.extend(batch_results if concurrency == 1 else batch_results)

        # === SUMMARY ===
        print("\n" + "=" * 80)
        print("SUMMARY BY CATEGORY")
        print("=" * 80)

        categories = sorted(set(r.category for r in all_results if not r.error))
        for cat in categories:
            cat_results = [r for r in all_results if r.category == cat and not r.error]
            if cat_results:
                avg_tps = mean(r.tps for r in cat_results)
                med_ttft = median(r.ttft for r in cat_results)
                print(f"  {cat:20s}  avg={avg_tps:.1f} tok/s  TTFT={med_ttft:.2f}s  n={len(cat_results)}")

        print("\n" + "=" * 80)
        print("SUMMARY BY OUTPUT LENGTH")
        print("=" * 80)
        # Group by approximate output length
        for target in [1024, 2048, 3072, 4096]:
            matching = [r for r in all_results if not r.error and abs(r.tokens - target) < target * 0.5]
            if matching:
                avg_tps = mean(r.tps for r in matching)
                print(f"  ~{target} tokens:  avg={avg_tps:.1f} tok/s  n={len(matching)}")

        # Save results
        output = [
            {
                "prompt_key": r.prompt_key,
                "category": r.category,
                "tokens": r.tokens,
                "time": round(r.time, 2),
                "ttft": round(r.ttft, 3),
                "tps": round(r.tps, 1),
                "error": r.error,
            }
            for r in all_results
        ]
        with open("/home/brandonmusic/klc-linux/benchmark_results/klc_real_world.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to ~/klc-linux/benchmark_results/klc_real_world.json")


if __name__ == "__main__":
    asyncio.run(main())
