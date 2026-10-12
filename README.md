# SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty

Official code for the paper [SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty](http://arxiv.org/abs/2610.13081).

GRPO-based reinforcement learning for [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with a difficulty-bucketed curriculum and self-rated compression reward. Includes the SFT data preparation pipeline and the MMAU evaluation script used in the paper.

## Links

- **Paper:** [arXiv:2610.13081](http://arxiv.org/abs/2610.13081)
- **Model:** [assasinatee/DiCoTA-Qwen2.5-Omni-7B](https://huggingface.co/assasinatee/DiCoTA-Qwen2.5-Omni-7B)
- **Dataset:** [assasinatee/DiCoTA](https://huggingface.co/datasets/assasinatee/DiCoTA)

## Install

```bash
pip install -r requirements.txt
```

## Data preparation

```bash
# SFT / RL non-overlapping split
python data_prep/split_sft_rl.py --compress_root $COMPRESS_ROOT --out data/split --ratio 0.7

# RL curriculum
python data_prep/build_bplus_curriculum.py \
    --compress-root $COMPRESS_ROOT \
    --include-ids data/split/rl_ids.jsonl \
    --out data/curriculum

# SFT sharegpt data
python sft/prepare_full_data.py \
    --compress_root $COMPRESS_ROOT --cota_root $COTA_ROOT \
    --output_dir converted_data \
    --s3_bucket $S3_BUCKET --s3_endpoint $S3_ENDPOINT

python sft/save_tokenized_data.py --data_dir converted_data --media_dir $AUDIO_ROOT
```

SFT itself uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with the `qwen2_omni` template; point `dataset_dir` at `converted_data`.

## RL training

```bash
export BASE_MODEL=/path/to/Qwen2.5-Omni-7B
export SFT_CKPT=/path/to/sft/checkpoint
export AUDIO_ROOT=/path/to/CoTA/audio_files
export CURRICULUM=/path/to/curriculum_data.jsonl

bash scripts/train_grpo.sh          # torchrun, 8 GPUs
# or
torchrun --nproc_per_node=8 rl/train_grpo.py configs/grpo_choice.yaml
```

## Evaluation (MMAU)

```bash
python eval/eval_mmau_unified.py \
    --model $BASE_MODEL --ckpt $CKPT \
    --parquet /path/to/MMAU/test_mini.parquet \
    --out results/mmau.json
```

Option accuracy with exact duplicate choices treated as equivalent.

## License

Apache License 2.0.

## Citation

```bibtex
@article{li2026swear,
  title   = {SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty},
  author  = {Li, Yixuan and Lv, Minghao and Xu, Xuenan and Zheng, Zihao and Wu, Mengyue},
  journal = {arXiv preprint arXiv:2610.13081},
  year    = {2026}
}
```
