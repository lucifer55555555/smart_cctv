"""
patch_model.py — One-time fix for the frozen violence_cnn_lstm.pt model.

ROOT CAUSE:
  The TorchScript model internally uses `x.view(b, t, -1)` after the CNN
  backbone. When the CNN output tensor is non-contiguous (common with pooling /
  global-average-pool layers), PyTorch raises:
      RuntimeError: view size is not compatible with … Use .reshape(...) instead.

FIX:
  Walk every node in the TorchScript computation graph (recursively through all
  sub-blocks) and replace every `aten::view` with `aten::reshape`.
  `reshape` is identical to `view` when the tensor IS contiguous, and falls
  back to a copy when it is not — so it is always safe.

USAGE:
  python patch_model.py
  (Run once; the fixed file overwrites the original.)
"""

import sys
import os
import shutil
import torch

SRC_MODEL = os.path.join(os.path.dirname(__file__), "models", "violence_cnn_lstm.pt")
BAK_MODEL = os.path.join(os.path.dirname(__file__), "models", "violence_cnn_lstm.pt.bak")


# ---------------------------------------------------------------------------
# Graph-level IR patch
# ---------------------------------------------------------------------------

def _fix_graph(graph) -> int:
    """
    Recursively replace every ``aten::view`` node with ``aten::reshape``.
    Returns the total number of replacements made.
    """
    replacements = 0

    # We must restart iteration each time we mutate the graph.
    changed = True
    while changed:
        changed = False
        for node in graph.nodes():
            # Recurse into sub-blocks first (e.g., prim::If / prim::Loop)
            for block in node.blocks():
                replacements += _fix_graph(block)

            if node.kind() == "aten::view":
                inputs = list(node.inputs())
                
                # Create the new reshape node
                new_node = graph.create("aten::reshape", inputs)
                new_node.output().setType(node.output().type())
                
                # Insert it right after the view node
                new_node.insertAfter(node)

                # Redirect all consumers of the old output to the new node
                node.output().replaceAllUsesWith(new_node.output())
                node.destroy()

                replacements += 1
                changed = True   # restart: graph iterator is now invalid
                break            # break the for-loop, outer while will retry

    return replacements


def patch_model(src_path: str, bak_path: str) -> None:
    if not os.path.exists(src_path):
        print(f"[PATCH] ERROR: Model not found at {src_path}")
        sys.exit(1)

    # --- Backup original ---
    if not os.path.exists(bak_path):
        print(f"[PATCH] Backing up original model → {bak_path}")
        shutil.copy2(src_path, bak_path)
    else:
        print(f"[PATCH] Backup already exists: {bak_path}  (skipping backup)")

    # --- Load ---
    print(f"[PATCH] Loading TorchScript model from {src_path} …")
    model = torch.jit.load(src_path, map_location="cpu")

    # --- Inline sub-graphs so nested view calls become visible at top level ---
    print("[PATCH] Inlining sub-graphs …")
    torch._C._jit_pass_inline(model.graph)

    # --- Replace view → reshape ---
    print("[PATCH] Replacing aten::view → aten::reshape …")
    n = _fix_graph(model.graph)
    print(f"[PATCH] Replaced {n} view node(s).")

    if n == 0:
        print("[PATCH] WARNING: No view nodes found. The model may already be fixed,")
        print("        or the inline pass changed the graph structure. Saving anyway.")

    # --- Verify the fix: a dummy forward pass ---
    print("[PATCH] Verifying fix with a dummy forward pass …")
    seq_len = 30          # must match the model's expected sequence length
    try:
        dummy = torch.zeros(1, seq_len, 3, 224, 224)  # (B, T, C, H, W)
        with torch.no_grad():
            out = model(dummy)
        prob = torch.sigmoid(out).item()
        print(f"[PATCH] ✅ Verification passed! dummy_prob={prob:.4f}")
    except Exception as e:
        print(f"[PATCH] ⚠️  Verification FAILED: {e}")
        print("       The model may need retraining. Check if the architecture")
        print("       uses operations other than view/reshape that cause issues.")

    # --- Save (overwrite original with fixed version) ---
    print(f"[PATCH] Saving fixed model → {src_path}")
    torch.jit.save(model, src_path)
    print("[PATCH] Done! The violence model has been patched in-place.")
    print(f"        Original backup kept at: {bak_path}")


if __name__ == "__main__":
    patch_model(SRC_MODEL, BAK_MODEL)
