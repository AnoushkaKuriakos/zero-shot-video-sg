"""VLM backend — supports Qwen2-VL and Qwen2.5-VL automatically."""
import numpy as np
from PIL import Image


def add_backend_args(ap):
    ap.add_argument('--backend', default='qwen', choices=['qwen', 'fake'])
    ap.add_argument('--load-4bit', action='store_true')
    ap.add_argument('--model-id', default='',
                    help='HF model id or local path; leave empty to auto-detect')


def backend_from_args(args, fake_replies=None):
    if args.backend == 'fake':
        return _FakeBackend(fake_replies or ['Fake reply.'])
    return _QwenBackend(args)


class _FakeBackend:
    def __init__(self, replies): self._r = replies; self._i = 0
    def chat(self, prompt, images=None, system=None, max_tokens=256, temperature=0.2):
        r = self._r[self._i % len(self._r)]; self._i += 1; return r


def _load_qwen(model_id, load_4bit=False):
    """Load whichever Qwen-VL variant is available."""
    import torch
    from transformers import AutoProcessor

    # Try Qwen2.5-VL first (newer), fall back to Qwen2-VL
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QVLM
        _cls_name = 'Qwen2.5-VL'
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QVLM
        _cls_name = 'Qwen2-VL'

    # Auto-detect model id if not given
    if not model_id:
        import os
        # Check common local paths first
        for cand in [
            os.path.expanduser(os.environ.get('QWEN_DIR', '~/models/Qwen2.5-VL-7B-Instruct')),
            os.path.expanduser('~/models/Qwen2-VL-7B-Instruct'),
        ]:
            if os.path.exists(os.path.join(cand, 'config.json')):
                model_id = cand
                break
        if not model_id:
            # Fall back to HF hub — try 2.5 first
            model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'

    print(f'[vlm_backends] loading {_cls_name} from {model_id} ...', flush=True)
    kw = dict(torch_dtype=torch.float16, device_map='auto')
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
    model = QVLM.from_pretrained(model_id, **kw)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f'[vlm_backends] {_cls_name} ready', flush=True)
    return model, processor


class _QwenBackend:
    def __init__(self, args):
        self._model, self._proc = _load_qwen(
            getattr(args, 'model_id', '') or '',
            getattr(args, 'load_4bit', False))

    def chat(self, prompt, images=None, system=None, max_tokens=256, temperature=0.2):
        import torch
        content = []
        for im in (images or []):
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im[..., ::-1].astype(np.uint8)
                                     if im.ndim == 3 and im.shape[2] == 3 else im)
            if not isinstance(im, Image.Image):
                im = Image.fromarray(np.array(im))
            content.append({'type': 'image', 'image': im})
        content.append({'type': 'text', 'text': prompt})
        msgs = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': content})
        text = self._proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        raw_imgs = [c['image'] for c in content if c['type'] == 'image']
        inputs = self._proc(
            text=[text], images=raw_imgs or None,
            return_tensors='pt').to(self._model.device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0)
        out = ids[:, inputs['input_ids'].shape[1]:]
        return self._proc.batch_decode(out, skip_special_tokens=True)[0].strip()
