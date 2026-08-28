# Pretrained Model Assets

Pretrained weights are not included in this repository because of their size
and licensing requirements. Download them from the official sources below and
place them under `models/` using the expected paths.

| Asset | Used by | Official source | Expected weight file | SHA-256 |
|---|---|---|---|---|
| GPT-2 Small | GPT4TS, UniTime, RPCL-TCNE-MTS-LLM | [Hugging Face](https://huggingface.co/openai-community/gpt2) | `gpt2/model.safetensors` | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| MOMENT-1-large | MOMENT | [Hugging Face](https://huggingface.co/AutonLab/MOMENT-1-large) | `MOMENT-1-large/model.safetensors` | `a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572` |
| Chronos-Bolt Base | Chronos | [Hugging Face](https://huggingface.co/amazon/chronos-bolt-base) | `chrones-bolt-base/model.safetensors` | `31f875483a3215bc6880a0837ea608a13ce55f88ad90538c3cdd0b29aeb60b36` |
| DADA | DADA | [Official repository](https://github.com/iambowen/DADA) | `DADA/pytorch_model.bin` | `66db3cc568d3df550bdb0aab7cf40c191acc1acd4bfee62b2b63af843369d850` |
| UniTS x32 | UniTS | [Official release](https://github.com/mims-harvard/UniTS/releases) | `UniTS/units_x32_pretrain_checkpoint.pth` | `455c81fdd269881cdde41706621156b1aafec37e6c6b3a63d2737793623297bd` |

The expected layout is:

```text
models/
├── gpt2/
├── MOMENT-1-large/
├── chrones-bolt-base/
├── DADA/
└── UniTS/
    └── units_x32_pretrain_checkpoint.pth
```

For DADA, retain `config.json`, `configuration_DADA.py`, and
`modeling_DADA.py` alongside `pytorch_model.bin`. For GPT-2, retain the model
configuration and tokenizer files alongside the weight file.

> `chrones-bolt-base` preserves the path used by the released experiment
> configurations; the upstream model name is `chronos-bolt-base`.

All benchmark configurations use local loading. GPT4TS and UniTime do not call
an online language-model API. Please follow the licenses and terms of the
upstream model providers.
