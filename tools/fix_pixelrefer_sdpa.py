"""Fix PixelRefer's DAMO-VL vision encoder so it runs WITHOUT flash-attn.

Upstream only tested flash_attention_2. Both fallbacks are broken:
  * VisionSdpaAttention.forward uses undefined `output_attentions`, `q_len`,
    `q` and `self.proj`   -> NameError("name 'output_attentions' ...")
  * VisionAttention (eager) uses undefined `q`  -> NameError("name 'q' ...")
and the vision tower is always loaded with 'sdpa' (encoder.py), so
--attn-impl in pixelrefer_caption.py never reaches it.

Usage: python tools/fix_pixelrefer_sdpa.py <PixelRefer repo root>   (makes a .bak)
Already applied to third_party/PixelRefer (re-running is a no-op;
a v1-patched file, which used an O(seq^2) dense mask, is upgraded from the .bak).
"""
import re, shutil, sys, pathlib

root = pathlib.Path(sys.argv[1]).expanduser()
p = root / 'pixelrefer/model/damovl_encoder/modeling_damovl_encoder.py'
s = p.read_text()
bak = pathlib.Path(str(p) + '.bak')
if '# PATCHED-SDPA-V2' in s:
    print('already patched'); sys.exit()
if '# PATCHED-SDPA' in s:            # v1 (dense mask): redo from the pristine copy
    s = bak.read_text()
else:
    shutil.copy(p, bak)

NEW_SDPA = '''class VisionSdpaAttention(VisionAttention):
    # PATCHED-SDPA-V2: upstream body referenced undefined names
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # each image/frame only attends to itself: one unmasked SDPA call per
        # cu_seqlens segment (what flash-attn varlen does). A dense
        # [seq, seq] block-diagonal mask is O(seq^2) -- ~28 GB for 16 frames.
        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_output = torch.empty_like(query_states)
        bounds = cu_seqlens.tolist()
        for a, b in zip(bounds[:-1], bounds[1:]):
            attn_output[:, a:b] = F.scaled_dot_product_attention(
                query_states[None, :, a:b], key_states[None, :, a:b],
                value_states[None, :, a:b], dropout_p=0.0)[0]
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        return self.out_proj(attn_output)


DAMOVL_VISION_ATTENTION_CLASSES = {'''

a = s.index('class VisionSdpaAttention(VisionAttention):')
b = s.index('DAMOVL_VISION_ATTENTION_CLASSES = {')
s = s[:a] + NEW_SDPA + s[b + len('DAMOVL_VISION_ATTENTION_CLASSES = {'):]

# eager path: `q` is undefined there too
s = s.replace('device=q.device, dtype=torch.bool', 'device=hidden_states.device, dtype=torch.bool')
s = s.replace('dtype=torch.float32).to(q.dtype)', 'dtype=torch.float32).to(query_states.dtype)')
# bool mask added to float scores is wrong in eager; turn it into an additive -inf mask
s = s.replace('attn_weights = attn_weights + attention_mask',
              'attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))')

if 'import torch.nn.functional as F' not in s and '\nimport torch.nn.functional as F' not in s:
    s = s.replace('import torch\n', 'import torch\nimport torch.nn.functional as F\n', 1)
p.write_text(s)
print('patched', p)
