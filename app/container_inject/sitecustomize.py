"""Injected into the DeepVariant container to enable bf16 inference.

WHY THIS EXISTS
---------------
Stock `google/deepvariant` ships an fp32 model. Intel AMX has no fp32 path --
it accelerates bf16 and int8 only. So simply raising the oneDNN ISA ceiling to
AVX512_CORE_AMX changes nothing: oneDNN reports an AMX-capable CPU, then runs
every convolution on `brgconv:avx512_core` anyway. Measured on a Xeon 6740P,
HG002 chr20, that gives a 0.98x "speedup" -- i.e. noise.

To put real work on the AMX tiles, TensorFlow's Grappler pass
`auto_mixed_precision_onednn_bfloat16` has to rewrite the graph so Conv2D and
MatMul consume bf16 tensors. Then oneDNN selects `brgconv:avx512_core_amx`.

This is the same mechanism Intel Labs' Open-Omics-DeepVariant uses. Their fork
calls it in-process; theirs is TF1 (`RewriterConfig`), DeepVariant 1.10 is TF2,
so the equivalent is `tf.config.optimizer.set_experimental_options`.

HOW IT GETS IN
--------------
Python imports `sitecustomize` automatically at interpreter startup if it is on
PYTHONPATH. DeepVariant's binaries are zipapps launched by a shell wrapper, so
there is no source file to patch and no supported flag to set. Mounting this
file read-only and setting PYTHONPATH is the least invasive option: the image
is unmodified and the change is one visible, auditable file.

SCOPE -- IMPORTANT
------------------
This only touches `call_variants`, the single deep-learning stage. An earlier
version applied to every Python process in the container, which meant all 192
`make_examples` shards imported TensorFlow at interpreter startup and applied a
graph-rewrite option they never use. That turned a 128-second chr20 run into
one still going after ten minutes. Keep the guard below.

HONESTY
-------
This is a PRECISION CHANGE, not a free lunch. bf16 has ~8 bits of mantissa
against fp32's 24. The demo therefore:
  - labels the comparison "AMX + bf16" versus "AVX-512 fp32", never "same work";
  - counts how many oneDNN primitives really used AMX and shows that number;
  - compares variant calls between the two legs and reports any difference.

WHY IT HOOKS THE IMPORT INSTEAD OF IMPORTING TENSORFLOW
-------------------------------------------------------
An earlier version simply did `import tensorflow` right here. That pulls
TensorFlow -- and its thread pools -- into the interpreter during startup,
before `call_variants` has done anything. On a full chr20 run (192 example
shards) that deadlocked hard: the process sat at 0% CPU with 389 threads all
parked in `futex_wait_queue_me`, forever. Creating TF's pools ahead of the
application's own process/pool setup is not safe.

So instead we install a meta-path finder that does nothing until the
application itself imports `tensorflow`, then flips the option the moment that
import finishes. TensorFlow is imported exactly once, at the time and in the
process the application chose.

Set DV_FORCE_BF16=0 (or leave it unset) and this file does nothing at all,
which is exactly what the AMX-OFF leg does.
"""

import os
import sys


def _is_call_variants() -> bool:
    """True only for the DeepVariant inference stage.

    make_examples and postprocess_variants do no dense DL compute, so rewriting
    their graphs buys nothing and costs a TensorFlow import in each of ~192
    shard processes.
    """
    return any("call_variants" in str(arg) for arg in sys.argv[:2])


if os.environ.get("DV_FORCE_BF16") == "1" and _is_call_variants():
    import importlib.abc
    import importlib.util

    def _enable_bf16(tf) -> None:
        try:
            tf.config.optimizer.set_experimental_options(
                {"auto_mixed_precision_onednn_bfloat16": True}
            )
            print(
                "[genomics-demo] bf16 auto-mixed-precision ON "
                "(Conv2D/MatMul -> bf16, enabling AMX tiles)",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - depends on the image
            # Never take the run down over this. A failure here means the leg
            # silently stays fp32, which the AMX dispatch counter will catch
            # and report rather than letting the demo claim an unearned win.
            print(f"[genomics-demo] bf16 injection FAILED: {exc}", flush=True)

    class _EnableBf16AfterTensorflow(importlib.abc.MetaPathFinder):
        """Flip the bf16 option the first time `tensorflow` is imported."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "tensorflow":
                return None
            # Step aside so the real finders resolve the module, and so this
            # hook fires only once.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec("tensorflow")
            except Exception as exc:  # pragma: no cover
                print(f"[genomics-demo] bf16 injection FAILED: {exc}", flush=True)
                return None
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            original_exec_module = loader.exec_module

            def exec_module(module):
                original_exec_module(module)
                _enable_bf16(module)

            try:
                loader.exec_module = exec_module
            except Exception as exc:  # pragma: no cover - immutable loader
                print(f"[genomics-demo] bf16 injection FAILED: {exc}", flush=True)
            return spec

    sys.meta_path.insert(0, _EnableBf16AfterTensorflow())
