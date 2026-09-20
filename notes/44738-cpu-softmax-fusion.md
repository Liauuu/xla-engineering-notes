# Avoid Splitting Softmax Scaled Logits Across CPU Fusions

Notes on OpenXLA PR #44738.

## Problem

In JAX eager mode, the softmax calculation produced the expected result. However, when the same computation was compiled with `jax.jit` on CPU, the output became `NaN`.

A simplified version of the computation looks like this:

```text
Q = ...
logits = Q * scale
row_max = reduce_max(logits)
shifted = logits - row_max
exp_shifted = exp(shifted)
weights = exp_shifted / reduce_sum(exp_shifted)
```

This is the usual numerically stable form of softmax.

Because `row_max` is the maximum value of `logits`,

```text
shifted = logits - row_max
```

should always satisfy

```text
shifted <= 0
```

and therefore

```text
exp(shifted) <= 1.
```

So `exp` should not overflow.

However, when I compared the intermediate values from eager and JIT execution, the first difference appeared at `shifted`.

Some values that should have been zero or negative became positive in the JIT version.

This caused the later computation to behave roughly like this:

```text
shifted > 0
    |
    v
exp(shifted)
    |
    v
   inf
    |
    v
inf / inf
    |
    v
   NaN
```

The final JIT output therefore contained `NaN`s even though the eager output was valid.

## Finding the Cause

To find which compiler optimization was causing the difference, I tried disabling different HLO passes.

When I disabled fusion with

```bash
XLA_FLAGS=--xla_disable_hlo_passes=fusion
```

the eager and JIT results matched again.

This pointed to XLA's instruction fusion pass.

Normally, instruction fusion tries to reduce memory traffic by putting operations into the same kernel. For cheap operations, it can sometimes be better to recompute a value inside multiple fused kernels instead of materializing it in memory and loading it again.

In many cases this is a useful optimization.

The problem here was that the value being recomputed was part of a numerically coupled calculation.

Conceptually, the original computation was:

```text
                 logits
                /      \
               /        \
              v          v
        reduce_max     subtract
              |          ^
              |          |
              +----------+
                         |
                         v
                        exp
```

where

```text
logits = dot * scale
```

is one logical value shared by both paths.

After CPU instruction fusion, the scaled logits could instead be recomputed independently in two different fusion kernels:

```text
                dot
               /   \
              /     \
             v       v

         Fusion A   Fusion B

         dot*scale  dot*scale
             |          |
             v          |
        reduce_max      |
             |          |
             +------> subtract
                        |
                        v
                       exp
```

This means the computation is no longer effectively

```text
logits - max(logits)
```

using exactly the same computed value.

Instead, it can behave more like

```text
logits_B - max(logits_A)
```

where `logits_A` and `logits_B` come from two separate recomputations.

Even though both kernels contain mathematically equivalent expressions, floating-point computation does not guarantee that the results are bit-identical.

Different fusion contexts can lead to different lowering decisions, including floating-point contraction such as FMA and different rounding behavior.

The difference can be extremely small, but for this calculation the relationship between the two values matters more than the absolute error.

For stable softmax, the important invariant is

```text
logits - max(logits) <= 0.
```

If `logits` is recomputed independently, that becomes

```text
logits_B - max(logits_A).
```

A small floating-point difference can therefore make a value that should be zero become slightly positive.

That small difference can then be amplified by `exp`, eventually producing `inf` and then `NaN` during normalization.

This was an important part of the bug for me: two computations can use the same mathematical expression but still produce slightly different floating-point values when they are lowered into different kernels.

## Fix

At first, I tried to fix the problem in `ShouldFuse`.

The reproducer contained a multiply for the scaled logits, so an early approach was to detect that multiply and prevent the problematic fusion.

However, this was not the right abstraction.

The real problem was not that the operation was a multiply, and it was not that fusion itself was unsafe.

The problem was that a value participating in this numerical relationship was being duplicated and recomputed independently.

Also, by the time this optimization runs on HLO, it is better not to depend on recognizing a high-level `softmax` function. The computation is represented as lower-level operations in the HLO graph.

So instead of looking specifically for `softmax` or for a `multiply`, the final fix looks for the computation pattern that creates the numerical dependency.

Conceptually, the pattern is:

```text
        elementwise producer
              |
          +---+---+
          |       |
          v       v
     reduce_max   |
          |       |
       broadcast  |
          |       |
          +----> subtract
                    |
                    v
                   exp
```

The matcher checks for an elementwise producer that is used by both:

```text
reduce_max(producer)
```

and the corresponding shifted exponential path:

```text
exp(producer - broadcast(reduce_max(producer)))
```

This makes the fix more general than matching only

```text
dot * scale
```

while still being narrow enough that unrelated computations are not affected.

For example, a similar graph using a sum reduction should not be protected:

```text
producer
   |
   +----> reduce_sum
   |
   +----> subtract -> exp
```

and a max/subtract computation that does not continue into `exp` should also continue to use the normal fusion behavior.

## Preventing Duplication

The final fix overrides `ComputeGloballyUnfusible`.

For producers matching the coupled reduction/shift/exp pattern, the instruction is added to the `do_not_duplicate` set.

Conceptually:

```text
if producer matches:

    producer
       |
       +----> reduce_max
       |
       +----> subtract -> exp

then:

    do_not_duplicate.insert(producer)
```

This is different from saying that the instruction cannot participate in fusion at all.

The goal is specifically to prevent this:

```text
Fusion A              Fusion B

recompute producer    recompute producer
       |                     |
   reduce_max              subtract
                              |
                              v
                             exp
```

and instead keep a single computed value that both paths can use:

```text
              producer
              /      \
             /        \
            v          v
      reduce_max     subtract
            |           ^
            +-----------+
                        |
                        v
                       exp
```

This preserves the numerical relationship between the value used to compute the maximum and the value used by the subtraction.

It also avoids disabling duplication too broadly.

Only producers matching the coupled max-reduction, subtract, and exponential structure are added to `do_not_duplicate`. Other cheap elementwise computations can still be duplicated by CPU fusion as before.

## Tests

I added positive tests for the patterns that should be protected and negative tests for similar-looking computations that should keep the existing fusion behavior.

The protected cases include the scaled-logits pattern and another elementwise producer feeding the same max-reduction and subtract/exp structure.

The negative cases are important because simply preventing duplication of every elementwise value with multiple consumers would be too broad and could unnecessarily reduce the effectiveness of CPU fusion.

For example, a sum reduction should still be handled normally:

```text
producer
   |
   +----> reduce_sum
   |
   +----> subtract -> exp
```

and a similar graph without the exponential path should also keep the normal behavior.

These tests check that the correctness fix only applies to the numerical dependency that caused the bug, rather than disabling useful fusion optimizations for unrelated computations.

I also reran the original JAX reproducer.

Before the fix, eager and JIT first diverged at `shifted`, eventually producing `inf` and `NaN`.

After the fix, the scaled logits were no longer independently recomputed across the two relevant fusion paths, and the eager and JIT intermediate values and final output matched.

## What I Learned

The main thing I learned from this bug was that floating-point correctness can depend on the relationship between multiple computations, not only on whether each individual expression is mathematically correct.

For stable softmax,

```text
x - max(x) <= 0
```

is an important invariant.

If the compiler independently recomputes `x` in different kernels, the computation can effectively become

```text
x2 - max(x1)
```

and floating-point differences between `x1` and `x2` can break that invariant.

I also learned that the right compiler fix is not always to disable the optimization where the bug becomes visible.

My first approach focused on `ShouldFuse`, but the more precise problem was duplication. Moving the fix to `ComputeGloballyUnfusible` allowed the fusion pass to keep doing its normal work while preventing recomputation only for producers participating in this coupled numerical pattern.

Finally, the negative tests were as important as the positive tests. A correctness fix in an optimization pass can easily become too conservative, so I needed to verify both that the problematic pattern was protected and that unrelated patterns could still be optimized normally.
